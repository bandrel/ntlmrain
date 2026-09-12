//! Synthetic GIDX0002/GRTB0002 table fixture for `tests/e2e.rs`.
//!
//! Nothing like this exists anywhere else in the repo. Every byte offset
//! below was derived by reading `Reader::open` (and the `BitReader`/page
//! decode helpers it calls) in `src/local_lookup.rs` directly -- see that
//! function for the authoritative cross-checks this fixture must satisfy.
//! The offset list right below mirrors the order `Reader::open` reads
//! them in.
//!
//! # Layout (one shard, one page, eight endpoints)
//!
//! GIDX0002 index file (256-byte header + shard directory + prefix
//! directory + low table):
//!
//! | field                | header offset | value                          |
//! |----------------------|---------------|--------------------------------|
//! | magic                | 0             | `b"GIDX0002"`                  |
//! | version              | 8             | 2                               |
//! | header_bytes         | 12            | 256                             |
//! | records              | 16            | 1024 (sizes `start_bits`, not the page's endpoint count -- see below) |
//! | parts                | 24            | 1                               |
//! | page_bytes           | 28            | 4096                            |
//! | start_bits           | 32            | 10 (= `bits_required(1023)`)    |
//! | rice_k               | 36            | 16                              |
//! | flags                | 40            | `0xff`                          |
//! | low_entry_bytes      | 44            | 4                               |
//! | blocks               | 48            | 1 (one page total)              |
//! | records_per_part     | 64            | 1024                            |
//! | min_endpoint         | 72            | 1000 (== first page endpoint)   |
//! | max_endpoint         | 80            | 8000 (== last page endpoint)    |
//! | fingerprint          | 96            | `FINGERPRINT` (shared w/ shard) |
//! | shards               | 132           | 1                               |
//! | shard_dir_offset     | 152           | 256 (== `TABLE_HEADER_BYTES`)   |
//! | prefix_offset        | 160           | 272 (= 256 + (shards+1)*8)      |
//! | low_offset           | 168           | 272 + (2^24+1)*4                |
//! | count_bytes          | 176           | 2                                |
//! | query_record_bytes   | 180           | 24                               |
//!
//! Total index file size is exactly `low_offset + blocks*4`, per
//! `Reader::open`'s `expected_index_bytes` check.
//!
//! `records` (1024) is *not* the number of endpoints in the one page this
//! fixture builds (8) -- it's the size of the abstract "start index"
//! collection that `start_bits` is sized for and that decoded `start`
//! values are bounds-checked against (`Reader::load_starts`:
//! `start >= self.info.records` is rejected). The two are independent by
//! design in the real format; this fixture picks 1024 (â‡’ `start_bits` =
//! 10, matching the plan's suggested small round number) purely so the
//! `start` values below (max 1000) fit comfortably under the bound.
//!
//! # Prefix directory (`(1<<24)+1` u32 entries, ~64 MiB)
//!
//! All of this fixture's endpoints have `endpoint >> 32 == 0`
//! ("high" = 0), so every page belongs to high-bucket 0: entry `0` is `0`
//! and every other entry (`1..=1<<24`) is `blocks` (`1`), which is the
//! standard "prefix-sum of page counts per high-24-bit bucket" encoding
//! `Reader::index_endpoint`/`Reader::lower_page` expect (see their
//! binary-search logic in `src/local_lookup.rs`). Built via an
//! exponential doubling `copy_within` fill (not a 16M-iteration Rust
//! loop) so construction stays fast -- see this file's timing note in the
//! task report.
//!
//! # GRTB0002 shard (one page, eight endpoints)
//!
//! Endpoints (ascending, all `high == 0`): `1000, 2000, .., 8000`.
//! Matching `start` values (each `< records` = 1024, arbitrary but
//! distinct): `3, 17, 42, 99, 200, 511, 777, 1000`.
//!
//! Page bytes: `u16` LE `count = 8`, then a `BitReader`-compatible
//! bitstream: seven Rice(k=16) deltas (`endpoints[i] - endpoints[i-1]`,
//! each exactly 1000 -- quotient 0, remainder 1000) followed by eight
//! `start_bits`-wide `start` values, zero-padded to fill the 4096-byte
//! page. The first (seed) endpoint (1000) is *not* delta-coded -- per
//! `Reader::decode_page`, page 0's first slot comes from the shard
//! header's `shard_min` field, not the bitstream.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const TABLE_HEADER_BYTES: usize = 256;
const PAGE_BYTES: usize = 4096;
const PREFIX_ENTRIES: usize = (1 << 24) + 1;
const FINGERPRINT: u64 = 0x1122_3344_5566_7788;

pub const RECORDS: u64 = 1024;
pub const START_BITS: u32 = 10;
pub const RICE_K: u32 = 16;
pub const BLOCKS: u64 = 1;
pub const SHARDS: u32 = 1;

/// A fully-built synthetic table on disk, kept alive for as long as this
/// value lives (the backing `tempfile::TempDir` is dropped -- and the
/// files deleted -- when this is dropped).
pub struct Fixture {
    pub _dir: tempfile::TempDir,
    pub data_base: PathBuf,
    pub index_path: PathBuf,
    /// The eight endpoints actually present on the fixture's one page,
    /// ascending.
    pub endpoints: Vec<u64>,
    /// `starts[i]` is the candidate `start` value for `endpoints[i]`.
    pub starts: Vec<u64>,
    /// How long building the ~64 MiB prefix directory took (see the task
    /// report for whether this warranted `#[ignore]`).
    pub prefix_build_time: std::time::Duration,
}

impl Fixture {
    /// An endpoint guaranteed to be outside `[min_endpoint, max_endpoint]`,
    /// so `LocalTable::lookup_endpoints` returns zero candidates for it.
    pub fn out_of_range_endpoint(&self) -> u64 {
        *self.endpoints.last().unwrap() + 1_000_000
    }
}

/// A minimal bit writer matching `BitReader`'s exact bit ordering in
/// `src/local_lookup.rs`: bits are packed LSB-first within each byte,
/// sequentially across bytes (bit `p` lives at byte `p/8`, bit `p%8` of
/// that byte). `read_bits`/`unary` reconstruct a little-endian bit stream
/// from that layout; writing bit-by-bit in the same order is the direct
/// inverse and needs no cleverness to get exactly right.
struct BitWriter {
    buf: Vec<u8>,
    pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        let byte_index = self.pos / 8;
        let bit_index = self.pos % 8;
        if byte_index >= self.buf.len() {
            self.buf.push(0);
        }
        if bit != 0 {
            self.buf[byte_index] |= 1 << bit_index;
        }
        self.pos += 1;
    }

    /// Writes the low `count` bits of `value`, LSB first (bit 0 of
    /// `value` goes to the current stream position, bit 1 to the next,
    /// ...), matching `BitReader::read_bits`.
    fn write_bits(&mut self, value: u64, count: u32) {
        for i in 0..count {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Writes `quotient` zero bits followed by a single one bit, matching
    /// `BitReader::unary` (which counts zero bits up to and consuming the
    /// terminating one bit).
    fn write_unary(&mut self, quotient: u64) {
        for _ in 0..quotient {
            self.write_bit(0);
        }
        self.write_bit(1);
    }
}

fn put_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Builds the ~64 MiB prefix directory: entry `0` is `0`, every other
/// entry (`1..=1<<24`) is `blocks`. Uses an exponential-doubling
/// `copy_within` fill instead of a 16M-iteration loop so this stays fast
/// regardless of build profile (debug vs. release) -- see this module's
/// doc comment.
fn build_prefix_table(blocks: u64) -> Vec<u8> {
    let mut buf = vec![0u8; PREFIX_ENTRIES * 4];
    // buf[0..4] (entry 0) stays zero. Fill buf[4..] with the repeating
    // 4-byte `blocks` pattern via doubling.
    let pattern = (blocks as u32).to_le_bytes();
    buf[4..8].copy_from_slice(&pattern);
    let mut filled = 4usize; // bytes filled starting at offset 4
    let region_len = buf.len() - 4;
    while filled < region_len {
        let copy_len = filled.min(region_len - filled);
        buf.copy_within(4..4 + copy_len, 4 + filled);
        filled += copy_len;
    }
    buf
}

/// Builds the one 4096-byte page: `u16` LE `count`, then the delta +
/// start bitstream, zero-padded to `PAGE_BYTES`.
fn build_page(endpoints: &[u64], starts: &[u64]) -> [u8; PAGE_BYTES] {
    assert_eq!(endpoints.len(), starts.len());
    let count = endpoints.len();
    assert!(count > 0 && count <= 1024);

    let mut writer = BitWriter::new();
    // Deltas for slots 1..count (slot 0 comes from the shard header's
    // `shard_min` seed, not the bitstream -- see `Reader::decode_page`).
    for window in endpoints.windows(2) {
        let delta = window[1] - window[0];
        let quotient = delta >> RICE_K;
        let remainder = delta & ((1u64 << RICE_K) - 1);
        writer.write_unary(quotient);
        writer.write_bits(remainder, RICE_K);
    }
    // Starts, in the same order as `endpoints`.
    for &start in starts {
        writer.write_bits(start, START_BITS);
    }

    let mut page = [0u8; PAGE_BYTES];
    page[0..2].copy_from_slice(&(count as u16).to_le_bytes());
    let body = &mut page[2..];
    assert!(
        writer.buf.len() <= body.len(),
        "fixture page bitstream ({} bytes) overflowed the {}-byte page body",
        writer.buf.len(),
        body.len()
    );
    body[..writer.buf.len()].copy_from_slice(&writer.buf);
    page
}

/// Builds the fixture into a fresh `tempfile::tempdir()`. Never call this
/// in a hot loop -- the ~64 MiB prefix directory write dominates
/// construction time (see `Fixture::prefix_build_time` / the task
/// report).
pub fn build_fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir for synthetic table fixture");
    let data_base = dir.path().join("table");
    let index_path = dir.path().join("table.gidx");

    let endpoints: Vec<u64> = vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
    let starts: Vec<u64> = vec![3, 17, 42, 99, 200, 511, 777, 1000];
    let min_endpoint = *endpoints.first().unwrap();
    let max_endpoint = *endpoints.last().unwrap();

    let shard_dir_offset: u64 = TABLE_HEADER_BYTES as u64;
    let prefix_offset: u64 = shard_dir_offset + (SHARDS as u64 + 1) * 8;
    let prefix_start = Instant::now();
    let prefix_table = build_prefix_table(BLOCKS);
    let prefix_build_time = prefix_start.elapsed();
    assert_eq!(prefix_table.len(), PREFIX_ENTRIES * 4);
    let low_offset: u64 = prefix_offset + prefix_table.len() as u64;
    let index_bytes: u64 = low_offset + BLOCKS * 4;

    // --- GIDX0002 header ---
    let mut header = [0u8; TABLE_HEADER_BYTES];
    header[0..8].copy_from_slice(b"GIDX0002");
    put_u32(&mut header, 8, 2); // version
    put_u32(&mut header, 12, TABLE_HEADER_BYTES as u32); // header_bytes
    put_u64(&mut header, 16, RECORDS); // records
    put_u32(&mut header, 24, 1); // parts
    put_u32(&mut header, 28, PAGE_BYTES as u32); // page_bytes
    put_u32(&mut header, 32, START_BITS); // start_bits
    put_u32(&mut header, 36, RICE_K); // rice_k
    put_u32(&mut header, 40, 0xff); // flags
    put_u32(&mut header, 44, 4); // low table entry size (bytes)
    put_u64(&mut header, 48, BLOCKS); // blocks
    put_u64(&mut header, 64, RECORDS); // records_per_part
    put_u64(&mut header, 72, min_endpoint); // min_endpoint
    put_u64(&mut header, 80, max_endpoint); // max_endpoint
    put_u64(&mut header, 96, FINGERPRINT); // fingerprint
    put_u32(&mut header, 132, SHARDS); // shards
    put_u64(&mut header, 152, shard_dir_offset); // shard_dir_offset
    put_u64(&mut header, 160, prefix_offset); // prefix_offset
    put_u64(&mut header, 168, low_offset); // low_offset
    put_u32(&mut header, 176, 2); // count_bytes
    put_u32(&mut header, 180, 24); // query record bytes

    // --- shard directory: (shards+1) u64 page offsets ---
    let mut shard_dir = Vec::with_capacity((SHARDS as usize + 1) * 8);
    shard_dir.extend_from_slice(&0u64.to_le_bytes());
    shard_dir.extend_from_slice(&BLOCKS.to_le_bytes());

    // --- low table: one u32 per block (endpoint's low 32 bits) ---
    let mut low_table = Vec::with_capacity(BLOCKS as usize * 4);
    low_table.extend_from_slice(&(max_endpoint as u32).to_le_bytes());

    {
        let file = fs::File::create(&index_path).expect("create index file");
        let mut writer = BufWriter::new(file);
        writer.write_all(&header).unwrap();
        writer.write_all(&shard_dir).unwrap();
        writer.write_all(&prefix_table).unwrap();
        writer.write_all(&low_table).unwrap();
        writer.flush().unwrap();
    }
    let actual_len = fs::metadata(&index_path).unwrap().len();
    assert_eq!(actual_len, index_bytes, "index file size mismatch");

    // --- GRTB0002 shard header + page ---
    let page = build_page(&endpoints, &starts);
    let file_bytes: u64 = TABLE_HEADER_BYTES as u64 + BLOCKS * PAGE_BYTES as u64;

    let mut shard_header = [0u8; TABLE_HEADER_BYTES];
    shard_header[0..8].copy_from_slice(b"GRTB0002");
    put_u32(&mut shard_header, 8, 2); // version
    put_u64(&mut shard_header, 16, RECORDS); // records (matches index)
    put_u32(&mut shard_header, 32, START_BITS); // start_bits
    put_u32(&mut shard_header, 36, RICE_K); // rice_k
    put_u32(&mut shard_header, 40, 0xff); // flags
    put_u64(&mut shard_header, 48, BLOCKS); // shard_pages
    put_u64(&mut shard_header, 56, file_bytes); // file_bytes
    put_u64(&mut shard_header, 72, min_endpoint); // shard_min
    put_u64(&mut shard_header, 96, FINGERPRINT); // fingerprint (matches index)
    put_u32(&mut shard_header, 128, 0); // shard index
    put_u32(&mut shard_header, 132, SHARDS); // shards
    put_u64(&mut shard_header, 136, 0); // records seen before this shard
    put_u64(&mut shard_header, 144, RECORDS); // records contributed by this shard

    let shard_path = data_base_shard_path(&data_base, 0);
    {
        let file = fs::File::create(&shard_path).expect("create shard file");
        let mut writer = BufWriter::new(file);
        writer.write_all(&shard_header).unwrap();
        writer.write_all(&page).unwrap();
        writer.flush().unwrap();
    }
    let actual_shard_len = fs::metadata(&shard_path).unwrap().len();
    assert_eq!(actual_shard_len, file_bytes, "shard file size mismatch");

    Fixture {
        _dir: dir,
        data_base,
        index_path,
        endpoints,
        starts,
        prefix_build_time,
    }
}

/// Mirrors `src/local_lookup.rs`'s private `shard_path` helper: shards
/// are `<base-without-.grtb>.<shard:04>.grtb`.
fn data_base_shard_path(base: &std::path::Path, shard: u32) -> PathBuf {
    let raw = base.to_string_lossy();
    let stem = raw.strip_suffix(".grtb").unwrap_or(&raw);
    PathBuf::from(format!("{stem}.{shard:04}.grtb"))
}

/// Encodes an `NTLMEND1` endpoint-file blob for the given endpoints, in
/// the format `ntlmrain::local_lookup::parse_endpoint_file` expects (see
/// that function).
pub fn encode_endpoint_file(endpoints: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32 + endpoints.len() * 8);
    buf.extend_from_slice(b"NTLMEND1");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&8u32.to_le_bytes()); // record size
    buf.extend_from_slice(&(endpoints.len() as u64).to_le_bytes()); // count
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for endpoint in endpoints {
        buf.extend_from_slice(&endpoint.to_le_bytes());
    }
    buf
}
