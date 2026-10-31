//! Device↔host transfers with **one** round trip per direction per layer.
//!
//! The expert lane is host-resident (`-cmoe`), so every block hands the MoE
//! activation down to the CPU and takes the FFN result back up. With one
//! `vkQueueSubmit` + `vkWaitForFences` per transfer that would be ~10 round trips
//! per block; this module keeps one persistent host-visible staging buffer and
//! packs every copy of a phase into a single command buffer, so a block costs
//! exactly one read and one write.
//!
//! The measured rates this is built against: RX 6950 XT PCIe H2D 28.25–28.71
//! GB/s (`docs/phase-2/01`), Radeon VII PCIe 4.0 ×4 5.4–6.6 GB/s (2026-10-29). At these sizes
//! (10 KiB) latency dominates bandwidth, which is why the count of round trips
//! — not their size — is what this module minimises.

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Qc};

use breccia_kernels::GemvTiming;

/// One persistent host-visible staging buffer, used for every transfer.
///
/// Deliberately *not* one buffer per call: `breccia_kernels::gemv::upload`
/// allocates and frees a staging buffer per tensor, which is right for a load
/// and wrong for a per-layer round trip.
pub struct Staging {
    buf: Buf,
    /// The buffer, mapped once for the whole session.
    ///
    /// `Gpu::read_host`/`write_host` map and unmap on every call, which measures
    /// as ~100 us of the ~150 us a 40 KiB round trip costs here — more than the
    /// transfer itself. A staging buffer never changes its memory type, so the
    /// mapping can be created once and used until the buffer is destroyed.
    ptr: *mut u8,
    cap: u64,
    /// Bytes reserved at the head for in-batch readbacks ([`block_region`]).
    region: u64,
    /// Bytes moved and round trips taken, for the report.
    pub bytes_down: u64,
    pub bytes_up: u64,
    pub trips_down: u64,
    pub trips_up: u64,
    pub seconds_down: f64,
    pub seconds_up: f64,
}

/// Bytes at the head of every staging buffer that [`Staging::read_batch`] never
/// touches, so a block's readback can ride **inside the block's own batch**
/// (`breccia_kernels::batch_cmd::record_copy`) instead of paying a submission and
/// a fence of its own.
pub const BLOCK_REGION: u64 = 32 << 10;

/// Bytes a chunked session's staging buffer needs for the **PLE** phase's batched
/// readback: one residual stack, one key stack and one value stack per token.
///
/// `read_batch` refuses a total above the buffer's `cap`, and the PLE phase's whole
/// point is to read `t` tokens back in one call instead of `t` calls, so the cap has
/// to be a construction parameter rather than the ~1 MiB a single readback needs
/// (`docs/phase-4` §8z/§8ai). This is the function a test can check without a GPU.
pub fn ple_stage_bytes(tokens: usize, hc_dim: usize, n_embd: usize) -> u64 {
    let per_token = (2 * hc_dim + n_embd) as u64 * 4;
    (tokens as u64).max(1) * per_token + 4096
}

/// Bytes reserved at the head of a staging buffer for the in-batch readback: one
/// [`BLOCK_REGION`] per token, so a whole prefill chunk's lane inputs fit.
pub fn block_region(tokens: usize) -> u64 {
    BLOCK_REGION * if tokens == 0 { 1 } else { tokens as u64 }
}

impl Staging {
    /// Allocate a staging buffer of `cap` bytes (rounded up to 4).
    pub fn new(gpu: &Gpu, cap: u64) -> Result<Self> {
        Self::with_block_region(gpu, cap, 1)
    }

    /// [`Staging::new`] with room for `tokens` in-batch readback slots.
    pub fn with_block_region(gpu: &Gpu, cap: u64, tokens: usize) -> Result<Self> {
        let cap = cap.max(64);
        let region = block_region(tokens);
        let size = cap + region + breccia_kernels::gemv::TAIL_PAD;
        let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // §11f: `BRECCIA_STAGING_CACHED=1` prefers a HOST_CACHED type, because the
        // CPU reads uncached (write-combined) staging at ~0.3 GB/s in `read_batch`.
        let pref = if std::env::var("BRECCIA_STAGING_CACHED").as_deref() == Ok("1") {
            hv | vk::MemoryPropertyFlags::HOST_CACHED
        } else {
            hv
        };
        let buf = gpu
            .buf(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                hv,
                pref,
            )
            .map_err(|e| anyhow!("allocating the {cap}-byte staging buffer: {e}"))?;
        // SAFETY: `buf` is a HOST_VISIBLE | HOST_COHERENT allocation owned by this
        // struct, and the mapping is created once and released in `release`
        // before the buffer is destroyed.
        let ptr = unsafe {
            gpu.device
                .map_memory(buf.mem, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow!("mapping the {cap}-byte staging buffer: {e}"))?
                as *mut u8
        };
        Ok(Staging {
            buf,
            ptr,
            cap,
            region,
            bytes_down: 0,
            bytes_up: 0,
            trips_down: 0,
            trips_up: 0,
            seconds_down: 0.0,
            seconds_up: 0.0,
        })
    }

    pub fn capacity(&self) -> u64 {
        self.cap
    }
    /// Bytes the staging buffer occupies (for the VRAM report).
    pub fn bytes(&self) -> u64 {
        self.buf.size
    }
    pub fn release(&self, gpu: &Gpu) {
        // SAFETY: the mapping came from this buffer's device memory and is
        // unmapped exactly once, before the memory is freed.
        unsafe { gpu.device.unmap_memory(self.buf.mem) };
        gpu.destroy_buf(&self.buf);
    }

    /// The buffer itself, for [`breccia_kernels::batch_cmd::record_copy`].
    pub fn buffer(&self) -> &Buf {
        &self.buf
    }

    /// Bytes the GPU wrote into the block region **inside an open batch** at
    /// `off` (relative to [`BLOCK_REGION`]); the caller has waited for that
    /// batch's segment, so the mapping is current.
    pub fn read_block(&self, off: u64, len: usize) -> &[u8] {
        assert!(
            off.saturating_add(len as u64) <= self.region,
            "read_block({off}, {len}) escapes the {}-byte block region",
            self.region
        );
        // SAFETY: bounds checked against the region, which is inside the mapping
        // (`cap + BLOCK_REGION + TAIL_PAD`), and the memory is host-coherent.
        unsafe { std::slice::from_raw_parts(self.ptr.add(off as usize), len) }
    }

    /// Device→host for a list of `(buffer, byte offset, length)` reads.
    ///
    /// All copies ride one command buffer and one fence wait; the returned
    /// vectors are in the order the items were given.
    pub fn read_batch(&mut self, gpu: &Gpu, items: &[(&Buf, u64, usize)]) -> Result<Vec<Vec<u8>>> {
        let total: u64 = items.iter().map(|(_, _, n)| *n as u64).sum();
        if total > self.cap {
            return Err(anyhow!(
                "read of {total} bytes exceeds the {}-byte staging buffer",
                self.cap
            ));
        }
        // Host-side bounds check: an out-of-range copy is a GPU page fault (a
        // dead device, and on the display card a dead machine), not a wrong
        // number. `docs/phase-3/06` §4.1 makes the same argument for `kv_append`.
        for (src, off, n) in items {
            if off.saturating_add(*n as u64) > src.size {
                return Err(anyhow!(
                    "read of {n} bytes at offset {off} is outside a {}-byte buffer",
                    src.size
                ));
            }
        }
        let t0 = std::time::Instant::now();
        let mut offs = Vec::with_capacity(items.len());
        // Past the block region: the in-batch copies own the head of the mapping.
        let mut at = self.region;
        for (_, _, n) in items {
            offs.push(at);
            at += *n as u64;
        }
        let regions: Vec<(vk::Buffer, vk::BufferCopy)> = items
            .iter()
            .zip(offs.iter())
            .map(|((src, off, n), dst_off)| {
                (
                    src.buf,
                    vk::BufferCopy::default()
                        .src_offset(*off)
                        .dst_offset(*dst_off)
                        .size(*n as u64),
                )
            })
            .collect();
        gpu.record(Qc::C, |d, cmd| unsafe {
            Gpu::barrier_all(d, cmd);
            for (src, region) in &regions {
                d.cmd_copy_buffer(cmd, *src, self.buf.buf, std::slice::from_ref(region));
            }
            Gpu::barrier_all(d, cmd);
        })
        .map_err(|e| anyhow!("recording the readback: {e}"))?;
        let fence = gpu
            .submit(Qc::C)
            .map_err(|e| anyhow!("submitting the readback: {e}"))?;
        gpu.wait_fence(fence)
            .map_err(|e| anyhow!("waiting for the readback: {e}"))?;

        let mut raw = vec![0u8; total as usize];
        // SAFETY: `read_batch` bounds-checked every `off + len` against the
        // buffer and `total <= cap`; the mapping covers
        // `cap + BLOCK_REGION + TAIL_PAD` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(self.region as usize),
                raw.as_mut_ptr(),
                total as usize,
            )
        };
        let base = self.region as usize;
        let out = items
            .iter()
            .zip(offs.iter())
            .map(|((_, _, n), off)| {
                let start = *off as usize - base;
                raw[start..start + n].to_vec()
            })
            .collect();
        self.bytes_down += total;
        self.trips_down += 1;
        self.seconds_down += t0.elapsed().as_secs_f64();
        Ok(out)
    }

    /// Host→device: write `data` into `dst` through the staging buffer.
    pub fn write(&mut self, gpu: &Gpu, dst: &Buf, data: &[u8]) -> Result<f64> {
        if data.len() as u64 > self.cap {
            return Err(anyhow!(
                "write of {} bytes exceeds the {}-byte staging buffer",
                data.len(),
                self.cap
            ));
        }
        if data.len() as u64 > dst.size {
            return Err(anyhow!(
                "write of {} bytes is outside a {}-byte buffer",
                data.len(),
                dst.size
            ));
        }
        let t0 = std::time::Instant::now();
        // SAFETY: `data.len() <= cap` was checked above and the mapping covers
        // `cap + BLOCK_REGION + TAIL_PAD` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.add(self.region as usize),
                data.len(),
            )
        };
        let region = vk::BufferCopy::default()
            .src_offset(self.region)
            .dst_offset(0)
            .size(data.len() as u64);
        // **No fence of its own when a batch is open** (§8bg). The copy only has to be
        // ordered *before the work that reads it*, and it rides the batch's submit:
        // host writes to the staging mapping are coherent before that submit, so a
        // submit+wait here (research 08: 33 us, of which 22-24 is the wait) buys
        // nothing. The wait survives for the one-shot path, where there is no batch.
        if breccia_kernels::batch_cmd::is_open() {
            breccia_kernels::batch_cmd::record_copy(
                gpu,
                dst,
                0,
                &self.buf,
                self.region,
                data.len() as u64,
            )
            .map_err(|e| anyhow!("recording the upload: {e}"))?;
            let secs = t0.elapsed().as_secs_f64();
            self.bytes_up += data.len() as u64;
            self.trips_up += 1;
            self.seconds_up += secs;
            return Ok(secs);
        }
        let src = self.buf.buf;
        let dstb = dst.buf;
        gpu.record(Qc::C, |d, cmd| unsafe {
            Gpu::barrier_all(d, cmd);
            d.cmd_copy_buffer(cmd, src, dstb, &[region]);
            Gpu::barrier_all(d, cmd);
        })
        .map_err(|e| anyhow!("recording the upload: {e}"))?;
        let fence = gpu
            .submit(Qc::C)
            .map_err(|e| anyhow!("submitting the upload: {e}"))?;
        gpu.wait_fence(fence)
            .map_err(|e| anyhow!("waiting for the upload: {e}"))?;
        let secs = t0.elapsed().as_secs_f64();
        self.bytes_up += data.len() as u64;
        self.trips_up += 1;
        self.seconds_up += secs;
        Ok(secs)
    }

    /// Stage `data` in the mapping's upload area and record the host→device copy
    /// into the **open batch** instead of submitting one of its own.
    ///
    /// A chunked prefill's PLE phase is the caller (§8aj): its packed table rows
    /// are one concatenated block, and a `write` per chunk would put a
    /// submit+fence (~56 µs) in front of the batch that consumes it for no reason
    /// — `vkQueueSubmit` already orders prior host writes to coherent memory
    /// against the submitted work, so recording the copy is enough.
    ///
    /// The upload area is the head of the buffer past the block region; the caller
    /// must not [`Staging::read_batch`] until this batch has been waited for (the
    /// read lands in the same bytes, after the copy has been consumed).
    pub fn record_upload(&mut self, gpu: &Gpu, dst: &Buf, dst_off: u64, data: &[u8]) -> Result<()> {
        let len = data.len() as u64;
        if len > self.cap {
            return Err(anyhow!(
                "upload of {len} bytes exceeds the {}-byte staging buffer",
                self.cap
            ));
        }
        if dst_off.saturating_add(len) > dst.size {
            return Err(anyhow!(
                "upload of {len} bytes at offset {dst_off} is outside a {}-byte buffer",
                dst.size
            ));
        }
        // SAFETY: `len <= cap` was checked above and the mapping covers
        // `cap + BLOCK_REGION + TAIL_PAD` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.add(self.region as usize),
                data.len(),
            )
        };
        breccia_kernels::batch_cmd::record_copy(gpu, dst, dst_off, &self.buf, self.region, len)?;
        self.bytes_up += len;
        Ok(())
    }

    /// Host→device for several `(buffer, offset, bytes)` items: every block goes
    /// into the staging mapping, then one command buffer carries one copy per
    /// item behind a single fence.
    ///
    /// The chunked PLE phase writes `t` residuals back after its host block, and
    /// `Buf` has no offset field — so the items are distinct buffers and a
    /// per-buffer [`Staging::write`] would be `t` submissions (§8aj).
    pub fn write_regions(&mut self, gpu: &Gpu, items: &[(&Buf, u64, &[u8])]) -> Result<f64> {
        if items.is_empty() {
            return Ok(0.0);
        }
        let total: u64 = items.iter().map(|(_, _, d)| d.len() as u64).sum();
        if total > self.cap {
            return Err(anyhow!(
                "write of {total} bytes exceeds the {}-byte staging buffer",
                self.cap
            ));
        }
        for (dst, off, data) in items {
            if off.saturating_add(data.len() as u64) > dst.size {
                return Err(anyhow!(
                    "write of {} bytes at offset {off} is outside a {}-byte buffer",
                    data.len(),
                    dst.size
                ));
            }
        }
        let t0 = std::time::Instant::now();
        let mut regions: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::with_capacity(items.len());
        let mut at = self.region;
        for (dst, off, data) in items {
            // SAFETY: the blocks tile `total <= cap`, checked above, and the
            // mapping covers `cap + BLOCK_REGION + TAIL_PAD` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(at as usize), data.len())
            };
            regions.push((
                dst.buf,
                vk::BufferCopy::default()
                    .src_offset(at)
                    .dst_offset(*off)
                    .size(data.len() as u64),
            ));
            at += data.len() as u64;
        }
        let src = self.buf.buf;
        gpu.record(Qc::C, |d, cmd| unsafe {
            Gpu::barrier_all(d, cmd);
            for (dst, region) in &regions {
                d.cmd_copy_buffer(cmd, src, *dst, std::slice::from_ref(region));
            }
            Gpu::barrier_all(d, cmd);
        })
        .map_err(|e| anyhow!("recording the writeback: {e}"))?;
        let fence = gpu
            .submit(Qc::C)
            .map_err(|e| anyhow!("submitting the writeback: {e}"))?;
        gpu.wait_fence(fence)
            .map_err(|e| anyhow!("waiting for the writeback: {e}"))?;
        let secs = t0.elapsed().as_secs_f64();
        self.bytes_up += total;
        self.trips_up += 1;
        self.seconds_up += secs;
        Ok(secs)
    }

    /// Read a single `f32` vector back (token logits, router weights, …).
    pub fn read_f32(&mut self, gpu: &Gpu, src: &Buf, n: usize) -> Result<Vec<f32>> {
        let v = self.read_batch(gpu, &[(src, 0, n * 4)])?;
        Ok(breccia_kernels::ops::f32_from_bytes(&v[0]))
    }

    /// Summary of the traffic this buffer carried, for the report.
    pub fn totals(&self) -> GemvTiming {
        GemvTiming {
            per_iter_s: self.seconds_down + self.seconds_up,
            iters: (self.trips_down + self.trips_up) as u32,
            total_s: self.bytes_down as f64 + self.bytes_up as f64,
            used_timestamps: false,
        }
    }
}

/// A device buffer that is **permanently mapped** into host memory, for the one
/// vector a block hands down to the host and takes straight back.
///
/// The MoE result has to travel host→device once per block. Going through
/// [`Staging`] costs a record + submit + `vkWaitForFences` per block (~180 µs
/// here, 48 times a token); writing straight into a coherent mapping costs a
/// `memcpy`, because `vkQueueSubmit` already orders prior host writes to coherent
/// memory against the submitted work — and the next thing that reads this buffer
/// is the ffn `HC_WRITE`, which is the next submission.
pub struct Mailbox {
    pub buf: Buf,
    ptr: *mut u8,
    size: u64,
}

impl Mailbox {
    pub fn new(gpu: &Gpu, size: u64) -> Result<Self> {
        let size = size.max(16);
        let buf = gpu
            .buf(
                size + breccia_kernels::gemv::TAIL_PAD,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|e| anyhow!("allocating the {size}-byte mailbox: {e}"))?;
        // SAFETY: host-visible, host-coherent memory owned by this struct; the
        // mapping outlives every use and is released before the buffer.
        let ptr = unsafe {
            gpu.device
                .map_memory(buf.mem, 0, buf.size, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow!("mapping the mailbox: {e}"))? as *mut u8
        };
        Ok(Mailbox { buf, ptr, size })
    }

    /// Copy `data` into the mapping. No driver call, no submission.
    pub fn write(&self, data: &[u8]) -> Result<()> {
        if data.len() as u64 > self.size {
            return Err(anyhow!(
                "mailbox write of {} bytes into {}-byte buffer",
                data.len(),
                self.size
            ));
        }
        // SAFETY: bounds checked above; the mapping covers `size + TAIL_PAD`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len()) };
        Ok(())
    }

    /// Copy `data` into the mapping at `off` — the streamed-expert path writes one
    /// expert's blobs into its slot without disturbing the others (§8x).
    pub fn write_at(&self, off: u64, data: &[u8]) -> Result<()> {
        if off.saturating_add(data.len() as u64) > self.size {
            return Err(anyhow!(
                "mailbox write of {} bytes at {off} escapes a {}-byte buffer",
                data.len(),
                self.size
            ));
        }
        // SAFETY: bounds checked above; the mapping covers `size + TAIL_PAD`.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(off as usize), data.len());
        }
        Ok(())
    }

    /// Copy the mapping out into `out`. No driver call, no submission — the
    /// caller must have waited for the batch that wrote it (the chunked prefill
    /// reads the chunk's router ids this way once per block, §8v).
    pub fn read(&self, out: &mut [u8]) -> Result<()> {
        if out.len() as u64 > self.size {
            return Err(anyhow!(
                "mailbox read of {} bytes from a {}-byte buffer",
                out.len(),
                self.size
            ));
        }
        // SAFETY: bounds checked above; the mapping covers `size + TAIL_PAD` and
        // the memory is host-coherent.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    pub fn bytes(&self) -> u64 {
        self.buf.size
    }

    pub fn release(&self, gpu: &Gpu) {
        // SAFETY: unmapped exactly once, before the memory is freed.
        unsafe { gpu.device.unmap_memory(self.buf.mem) };
        gpu.destroy_buf(&self.buf);
    }
}

#[cfg(test)]
mod tests {
    /// The staging math is pure: `read_batch` lays items out back to back and the
    /// offsets must tile `total` exactly (a mismatch would silently hand a caller
    /// another buffer's bytes).
    /// The PLE phase's batched readback needs `t * (2*hc_dim + n_embd) * 4` bytes and
    /// `read_batch` refuses a total above `cap`, so the chunked session must size its
    /// staging buffer from this, not from the single-readback default (§8ai).
    #[test]
    fn ple_stage_bytes_covers_a_whole_chunk() {
        let (hc_dim, n_embd) = (10240usize, 2560usize);
        for t in [1usize, 8, 64, 128] {
            let want = t as u64 * (2 * hc_dim + n_embd) as u64 * 4;
            let got = super::ple_stage_bytes(t, hc_dim, n_embd);
            assert!(
                got >= want,
                "t = {t}: {got} B does not cover a {want} B readback"
            );
            // and it must not be wildly oversized either
            assert!(got <= want + 65536, "t = {t}: {got} B is oversized");
        }
        assert!(super::ple_stage_bytes(128, hc_dim, n_embd) > 10 << 20);
    }

    #[test]
    fn read_offsets_tile_the_staging_buffer() {
        let items = [(0u64, 4usize), (4, 12), (16, 40)];
        let mut at = 0u64;
        let mut offs = Vec::new();
        for (_, n) in items {
            offs.push(at);
            at += n as u64;
        }
        assert_eq!(offs, vec![0, 4, 16]);
        assert_eq!(at, 56);
    }
}

/// §11j item 5: one anonymous host allocation imported (`VK_EXT_external_memory_host`)
/// on each card a dual-card ship touches, so the hand-off is two DMAs and no CPU
/// copy (the staged path is DMA, memcpy out, memcpy in, DMA). On by default (Carl 10-30);
/// `BRECCIA_SHIP_IMPORT=0` opts out; never one card's exported allocation.
pub struct ShipImport {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    pub cap: u64,
    bufs: Vec<Option<Buf>>,
}

impl ShipImport {
    /// `cap` is rounded up to 64 KiB so the buffer's `allocation_size` stays inside
    /// the host range (both cards report a 4 KiB import alignment).
    pub fn new(cap: u64, n_devices: usize) -> Result<Self> {
        const GRAIN: u64 = 64 << 10;
        let cap = cap.max(1).div_ceil(GRAIN) * GRAIN;
        let layout = std::alloc::Layout::from_size_align(cap as usize, GRAIN as usize)?;
        // SAFETY: non-zero size, valid power-of-two alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(anyhow!("ship import alloc of {cap} bytes failed"));
        }
        Ok(Self { ptr, layout, cap, bufs: (0..n_devices).map(|_| None).collect() })
    }

    /// Import the range on device `i` if it is not already.
    pub fn ensure(&mut self, i: usize, gpu: &Gpu) -> Result<()> {
        if self.bufs[i].is_some() {
            return Ok(());
        }
        if !gpu.external_memory_host {
            return Err(anyhow!("device {i} lacks VK_EXT_external_memory_host"));
        }
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let b = breccia_vk::transfer::import_from_host_ptr(gpu, self.ptr, self.cap, usage)
            .map_err(|e| anyhow!("device {i}: {e}"))?;
        self.bufs[i] = Some(b);
        Ok(())
    }

    pub fn get(&self, i: usize) -> &Buf {
        self.bufs[i].as_ref().expect("ShipImport::ensure ran for this device")
    }

    /// Destroy the imports; the host range is freed on drop, after this.
    pub fn release(&mut self, gpus: &[&Gpu]) {
        for (i, b) in self.bufs.iter_mut().enumerate() {
            if let Some(b) = b.take() {
                gpus[i].destroy_buf(&b);
            }
        }
    }
}

impl Drop for ShipImport {
    fn drop(&mut self) {
        // A live import still points here: leak rather than free under a device.
        if self.bufs.iter().any(Option::is_some) {
            return;
        }
        // SAFETY: same ptr/layout this allocation was made with.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// Fenced `(src, dst, src offset, dst offset, bytes)` copies on `gpu`, for the
/// [`ShipImport`] hand-off. The barriers add the HOST stage both ways: the other
/// side of the memory is another card's DMA, reusing the same addresses every ship.
pub fn copy_fenced(gpu: &Gpu, copies: &[(&Buf, &Buf, u64, u64, u64)]) -> Result<f64> {
    copy_submit(gpu, copies)?.wait(gpu)
}

/// A [`copy_submit`] copy in flight on the device's one-shot transfer buffer: nothing
/// else may record on that buffer (`copy_*`, staged reads/writes) before [`Self::wait`].
pub struct PendingCopy {
    fence: vk::Fence,
    t0: std::time::Instant,
}

impl PendingCopy {
    pub fn wait(self, gpu: &Gpu) -> Result<f64> {
        gpu.wait_fence(self.fence).map_err(|e| anyhow!("waiting for the ship copy: {e}"))?;
        Ok(self.t0.elapsed().as_secs_f64())
    }
}

/// [`copy_fenced`] without the wait: the host is free to work while the DMA runs.
pub fn copy_submit(gpu: &Gpu, copies: &[(&Buf, &Buf, u64, u64, u64)]) -> Result<PendingCopy> {
    for (s, d, so, dof, n) in copies {
        if so.saturating_add(*n) > s.size || dof.saturating_add(*n) > d.size {
            return Err(anyhow!(
                "copy of {n} bytes ({so} of {}, {dof} of {}) is out of range",
                s.size,
                d.size
            ));
        }
    }
    let t0 = std::time::Instant::now();
    let rw = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    let host = vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE;
    let all = vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST;
    gpu.record(Qc::C, |d, cmd| unsafe {
        let pre = vk::MemoryBarrier::default().src_access_mask(rw | host).dst_access_mask(rw);
        d.cmd_pipeline_barrier(cmd, all, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[pre], &[], &[]);
        for (s, dst, so, dof, n) in copies {
            let r = vk::BufferCopy::default().src_offset(*so).dst_offset(*dof).size(*n);
            d.cmd_copy_buffer(cmd, s.buf, dst.buf, std::slice::from_ref(&r));
        }
        let post = vk::MemoryBarrier::default().src_access_mask(rw).dst_access_mask(rw | host);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, all, vk::DependencyFlags::empty(), &[post], &[], &[]);
    })
    .map_err(|e| anyhow!("recording the ship copy: {e}"))?;
    let fence = gpu.submit(Qc::C).map_err(|e| anyhow!("submitting the ship copy: {e}"))?;
    Ok(PendingCopy { fence, t0 })
}
