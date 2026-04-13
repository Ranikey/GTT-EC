import argparse
import logging
import sys
import typing as T
from pathlib import Path
from timeit import default_timer as timer

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def read_fasta(fasta_path: Path) -> T.List[T.Tuple[str, str]]:
    sequences = []
    with open(fasta_path, "r") as f:
        header, seq = "", []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            elif line:
                seq.append(line)
        if header:
            sequences.append((header, "".join(seq)))
    return sequences


def clean_header(raw_header: str) -> str:
    cleaned = raw_header.split("|")[0].strip()
    if not cleaned:
        raise ValueError(f"Cleaned header is empty! Original header: {raw_header}")
    return cleaned


def create_batched_sequence_datasest(
        sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)
    if batch_headers:
        yield batch_headers, batch_sequences


def get_pdbs_from_batched_outputs(outputs, input_lengths):
    pdbs = []
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)

    final_atom_positions = final_atom_positions.cpu().float().numpy()

    outputs_cpu = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                outputs_cpu[k] = v.cpu().float().numpy()
            else:
                outputs_cpu[k] = v.cpu().numpy()
        else:
            outputs_cpu[k] = v

    final_atom_mask = outputs_cpu["atom37_atom_exists"]

    for i, seq_len in enumerate(input_lengths):
        protein = OFProtein(
            aatype=outputs_cpu["aatype"][i][:seq_len],
            atom_positions=final_atom_positions[i][:seq_len],
            atom_mask=final_atom_mask[i][:seq_len],
            residue_index=outputs_cpu["residue_index"][i][:seq_len] + 1,
            b_factors=outputs_cpu["plddt"][i][:seq_len] * 100,
            chain_index=outputs_cpu["chain_index"][i][:seq_len] if "chain_index" in outputs_cpu else None,
        )
        pdbs.append(to_pdb(protein))
    return pdbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fasta", type=Path, required=True)
    parser.add_argument("-o", "--pdb", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-recycles", type=int, default=4)
    parser.add_argument("--max-tokens-per-batch", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {args.fasta}")

    if args.pdb.exists():
        logger.info(f"Output directory already exists: {args.pdb}")
    else:
        args.pdb.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output directory created: {args.pdb}")

    logger.info(f"Reading sequences from {args.fasta}")
    raw_sequences = read_fasta(args.fasta)
    cleaned_sequences = []
    for raw_header, seq in raw_sequences:
        cleaned_h = clean_header(raw_header)
        cleaned_sequences.append((cleaned_h, seq))

    all_sequences = sorted(cleaned_sequences, key=lambda header_seq: len(header_seq[1]))

    logger.info(f"Loaded {len(all_sequences)} sequences from {args.fasta}")

    logger.info("Loading tokenizer and model (Hugging Face Transformers)...")
    model_name = "facebook/esmfold_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)
    model = model.eval()

    if args.chunk_size is not None:
        model.trunk.set_chunk_size(args.chunk_size)

    device = f'cuda:{args.gpu}'
    if args.cpu_only:
        model = model.float().cpu()
    else:
        model = model.to(torch.bfloat16).to(device)

    logger.info("Starting Predictions")
    batched_sequences = create_batched_sequence_datasest(all_sequences, args.max_tokens_per_batch)

    num_completed = 0
    num_sequences = len(all_sequences)

    for headers, sequences in batched_sequences:
        pending_headers = []
        pending_sequences = []
        for h, s in zip(headers, sequences):
            expected_pdb_path = args.pdb / f"{h}.pdb"
            if expected_pdb_path.exists():
                logger.info(f"PDB file already exists, skipping: {expected_pdb_path}")
                continue
            pending_headers.append(h)
            pending_sequences.append(s)

        if not pending_headers:
            logger.info("All sequences in current batch have generated PDBs, skipping inference")
            continue

        headers = pending_headers
        sequences = pending_sequences
        input_lengths = [len(seq) for seq in sequences]

        start = timer()

        inputs = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if len(sequences) > 1:
                    logger.warning(f"OOM on batch size {len(sequences)}. Try lowering `--max-tokens-per-batch`.")
                else:
                    logger.warning(
                        f"OOM on sequence {headers[0]} (length {len(sequences[0])}). Try setting `--chunk-size 64`.")
                continue
            raise

        pdbs = get_pdbs_from_batched_outputs(outputs, input_lengths)

        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.2f}s"
        if len(sequences) > 1:
            time_string += f" (amortized, batch size {len(sequences)})"

        for i, (header, seq, pdb_string) in enumerate(zip(headers, sequences, pdbs)):
            mean_plddt = outputs.plddt[i][:input_lengths[i]].mean().item() * 100

            if outputs.ptm is not None:
                ptm = outputs.ptm[i].item() if outputs.ptm.ndim > 0 else outputs.ptm.item()
            else:
                ptm = 0.0

            output_file = args.pdb / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1

            logger.info(
                f"Predicted {header} (L={len(seq)}), pLDDT {mean_plddt:0.1f}, "
                f"pTM {ptm:0.3f} in {time_string}. "
                f"[{num_completed}/{num_sequences}]"
            )
