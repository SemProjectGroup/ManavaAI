"""
Memory-Efficient Merge Script for Large Dataset Chunks
Run this after the download to merge chunks without running out of memory
"""

import os
import gc
import pandas as pd
from tqdm import tqdm
import shutil

# Configuration - match your download settings
CONFIG = {
    'temp_dir': 'temp_chunks_20x',
    'output_dir': 'dataset_output_20x',
    'max_rows_per_file': 1_000_000,  # 1M rows per output file (smaller = less memory)
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def merge_chunks_streaming(chunk_type: str):
    """
    Merge chunks one at a time, writing output files incrementally.
    Never holds more than 2 chunks in memory at once.
    """
    print(f"\n{'='*60}")
    print(f"MERGING {chunk_type.upper()} CHUNKS (Memory-Efficient)")
    print(f"{'='*60}")
    
    # Get all chunk files for this type
    chunk_files = sorted([
        f for f in os.listdir(CONFIG['temp_dir']) 
        if f.startswith(f'{chunk_type}_') and f.endswith('.parquet')
    ])
    
    if not chunk_files:
        print(f"No {chunk_type} chunks found!")
        return 0
    
    print(f"Found {len(chunk_files)} {chunk_type} chunk files")
    
    total_rows = 0
    output_file_idx = 0
    current_buffer = []
    current_buffer_size = 0
    
    for chunk_file in tqdm(chunk_files, desc=f"Processing {chunk_type}"):
        try:
            # Load one chunk at a time
            chunk_path = os.path.join(CONFIG['temp_dir'], chunk_file)
            df = pd.read_parquet(chunk_path)
            
            # Add to buffer
            current_buffer.append(df)
            current_buffer_size += len(df)
            total_rows += len(df)
            
            # If buffer is large enough, write to file
            if current_buffer_size >= CONFIG['max_rows_per_file']:
                # Concatenate buffer
                combined = pd.concat(current_buffer, ignore_index=True)
                
                # Write in max_rows_per_file chunks
                while len(combined) >= CONFIG['max_rows_per_file']:
                    output_chunk = combined.iloc[:CONFIG['max_rows_per_file']]
                    output_path = os.path.join(
                        CONFIG['output_dir'], 
                        f'{chunk_type}_essays_part{output_file_idx:04d}.parquet'
                    )
                    output_chunk.to_parquet(output_path, index=False, compression='snappy')
                    print(f"  Wrote {output_path} ({len(output_chunk):,} rows)")
                    
                    combined = combined.iloc[CONFIG['max_rows_per_file']:]
                    output_file_idx += 1
                
                # Keep remainder in buffer
                if len(combined) > 0:
                    current_buffer = [combined]
                    current_buffer_size = len(combined)
                else:
                    current_buffer = []
                    current_buffer_size = 0
                
                # Force garbage collection
                del df
                gc.collect()
            
        except Exception as e:
            print(f"  Error processing {chunk_file}: {e}")
            continue
    
    # Write any remaining data
    if current_buffer:
        combined = pd.concat(current_buffer, ignore_index=True)
        
        # Split into appropriately sized files
        for i in range(0, len(combined), CONFIG['max_rows_per_file']):
            output_chunk = combined.iloc[i:i + CONFIG['max_rows_per_file']]
            output_path = os.path.join(
                CONFIG['output_dir'], 
                f'{chunk_type}_essays_part{output_file_idx:04d}.parquet'
            )
            output_chunk.to_parquet(output_path, index=False, compression='snappy')
            print(f"  Wrote {output_path} ({len(output_chunk):,} rows)")
            output_file_idx += 1
        
        del combined
        gc.collect()
    
    print(f"\n✓ {chunk_type.upper()}: {total_rows:,} total rows in {output_file_idx} files")
    return total_rows


def create_balanced_sample(sample_size: int = 5_000_000):
    """
    Create a balanced sample from the merged files.
    Samples equally from human and AI texts.
    """
    print(f"\n{'='*60}")
    print(f"CREATING BALANCED SAMPLE ({sample_size:,} per class)")
    print(f"{'='*60}")
    
    # Get output files
    human_files = sorted([
        f for f in os.listdir(CONFIG['output_dir']) 
        if f.startswith('human_') and f.endswith('.parquet')
    ])
    ai_files = sorted([
        f for f in os.listdir(CONFIG['output_dir']) 
        if f.startswith('ai_') and f.endswith('.parquet')
    ])
    
    # Count total rows in each
    human_total = 0
    for f in human_files:
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        human_total += len(df)
        del df
    
    ai_total = 0
    for f in ai_files:
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        ai_total += len(df)
        del df
    
    print(f"Total human texts: {human_total:,}")
    print(f"Total AI texts: {ai_total:,}")
    
    # Determine sample size per class
    per_class = min(sample_size, human_total, ai_total)
    print(f"Sampling {per_class:,} per class...")
    
    # Sample from human files
    print("\nSampling human texts...")
    human_samples = []
    samples_needed = per_class
    
    for f in tqdm(human_files):
        if samples_needed <= 0:
            break
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        # Calculate proportion to sample from this file
        take = min(len(df), samples_needed)
        if take < len(df):
            sample = df.sample(n=take, random_state=42)
        else:
            sample = df
        human_samples.append(sample)
        samples_needed -= len(sample)
        del df
        gc.collect()
    
    human_df = pd.concat(human_samples, ignore_index=True)
    del human_samples
    gc.collect()
    
    # Sample from AI files
    print("\nSampling AI texts...")
    ai_samples = []
    samples_needed = per_class
    
    for f in tqdm(ai_files):
        if samples_needed <= 0:
            break
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        take = min(len(df), samples_needed)
        if take < len(df):
            sample = df.sample(n=take, random_state=42)
        else:
            sample = df
        ai_samples.append(sample)
        samples_needed -= len(sample)
        del df
        gc.collect()
    
    ai_df = pd.concat(ai_samples, ignore_index=True)
    del ai_samples
    gc.collect()
    
    # Combine and shuffle
    print("\nCombining and shuffling...")
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    del human_df, ai_df
    gc.collect()
    
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save balanced sample
    output_path = os.path.join(CONFIG['output_dir'], 'balanced_sample.parquet')
    combined.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"\n✓ Saved balanced sample: {len(combined):,} texts to {output_path}")
    
    # Also save as CSV for easy inspection (first 100k only due to size)
    csv_sample = combined.head(100000)
    csv_path = os.path.join(CONFIG['output_dir'], 'balanced_sample_preview.csv')
    csv_sample.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV preview: {len(csv_sample):,} texts to {csv_path}")
    
    return len(combined)


def print_statistics():
    """Print comprehensive dataset statistics"""
    print(f"\n{'='*60}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*60}")
    
    human_count = 0
    ai_count = 0
    human_sources = {}
    ai_sources = {}
    
    for f in os.listdir(CONFIG['output_dir']):
        if not f.endswith('.parquet'):
            continue
        
        filepath = os.path.join(CONFIG['output_dir'], f)
        try:
            df = pd.read_parquet(filepath)
            
            if 'human' in f and 'balanced' not in f:
                human_count += len(df)
                if 'source' in df.columns:
                    for src, cnt in df['source'].value_counts().items():
                        human_sources[src] = human_sources.get(src, 0) + cnt
            elif 'ai' in f and 'balanced' not in f:
                ai_count += len(df)
                if 'source' in df.columns:
                    for src, cnt in df['source'].value_counts().items():
                        ai_sources[src] = ai_sources.get(src, 0) + cnt
            
            del df
            gc.collect()
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    
    print(f"\nHuman Texts: {human_count:,}")
    if human_sources:
        print("  Top sources:")
        for src, cnt in sorted(human_sources.items(), key=lambda x: -x[1])[:10]:
            print(f"    - {src}: {cnt:,}")
    
    print(f"\nAI Texts: {ai_count:,}")
    if ai_sources:
        print("  Top sources:")
        for src, cnt in sorted(ai_sources.items(), key=lambda x: -x[1])[:10]:
            print(f"    - {src}: {cnt:,}")
    
    print(f"\nTotal: {human_count + ai_count:,}")
    
    # Calculate total disk usage
    total_size = 0
    file_count = 0
    for f in os.listdir(CONFIG['output_dir']):
        if f.endswith('.parquet'):
            total_size += os.path.getsize(os.path.join(CONFIG['output_dir'], f))
            file_count += 1
    
    print(f"\nDisk Usage: {total_size / 1e9:.2f} GB across {file_count} files")


def cleanup_temp_chunks():
    """Optionally remove temporary chunk files after successful merge"""
    print(f"\n{'='*60}")
    print("CLEANUP")
    print(f"{'='*60}")
    
    response = input(f"Delete temporary chunks in '{CONFIG['temp_dir']}'? (y/n): ").strip().lower()
    if response == 'y':
        try:
            shutil.rmtree(CONFIG['temp_dir'])
            print("✓ Temporary chunks deleted")
        except Exception as e:
            print(f"✗ Error deleting temp files: {e}")
    else:
        print("Temporary chunks kept")


def main():
    print("="*60)
    print("MEMORY-EFFICIENT DATASET MERGER")
    print("="*60)
    
    # Check if temp directory exists
    if not os.path.exists(CONFIG['temp_dir']):
        print(f"Error: Temp directory '{CONFIG['temp_dir']}' not found!")
        print("Make sure you've run the download script first.")
        return
    
    # Count existing chunks
    human_chunks = len([f for f in os.listdir(CONFIG['temp_dir']) if f.startswith('human_')])
    ai_chunks = len([f for f in os.listdir(CONFIG['temp_dir']) if f.startswith('ai_')])
    print(f"\nFound {human_chunks} human chunks and {ai_chunks} AI chunks")
    
    # Merge human chunks
    human_total = merge_chunks_streaming('human')
    
    # Merge AI chunks
    ai_total = merge_chunks_streaming('ai')
    
    # Create balanced sample
    if human_total > 0 and ai_total > 0:
        create_balanced_sample(sample_size=5_000_000)
    
    # Print statistics
    print_statistics()
    
    # Optional cleanup
    cleanup_temp_chunks()
    
    print("\n" + "="*60)
    print("✓ MERGE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()