//! Dump the CPU reference dequantisation of packed blocks (stdin → stdout).
//!
//! Used by `tools/refcheck.py` to diff this crate's decode against the verbatim
//! `ggml-quants.c` reference: the C driver reads the same block bytes and writes
//! the same little-endian f32 stream.
//!
//! ```text
//! dequant_dump <TYPE> <nblocks>   # blocks on stdin, f32 on stdout
//! ```

use std::io::{Read, Write};

use breccia_kernels::reference;
use breccia_kernels::QuantType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dequant_dump <TYPE> <nblocks>");
        std::process::exit(2);
    }
    let ty: QuantType = args[1].parse()?;
    let n: usize = args[2].parse()?;
    let mut bytes = vec![0u8; n * ty.block_bytes()];
    std::io::stdin().read_exact(&mut bytes)?;

    let mut out = Vec::with_capacity(n * ty.block_elems() * 4);
    let mut vals = Vec::with_capacity(ty.block_elems());
    for b in 0..n {
        vals.clear();
        let off = b * ty.block_bytes();
        reference::dequant_block(ty, &bytes[off..off + ty.block_bytes()], &mut vals);
        for v in &vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::io::stdout().write_all(&out)?;
    Ok(())
}
