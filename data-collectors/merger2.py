"""
Ultra Memory-Efficient Merge - No Full Shuffle Required
"""

import os
import gc
import pandas as pd
from tqdm import tqdm
import random

CONFIG = {
    'temp_dir': 'temp_chunks_20x',
    'output_dir': 'dataset_output_20x',
    'max_rows_per_file': 500_000,  # 500K rows per file (safer)
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def merge_chunks_simple(chunk_type: str):
    """Merge chunks one at a time, writing immediately"""
    print(f"\n{'='*60}")
    print(f"MERGING {chunk_type.upper()} CHUNKS")
    print(f"{'='*60}")
    
    chunk_files = sorted([
        f for f in os.listdir(CONFIG['temp_dir']) 
        if f.startswith(f'{chunk_type}_') and f.endswith('.parquet')
    ])
    
    if not chunk_files:
        print(f"No {chunk_type} chunks found!")
        return 0
    
    print(f"Found {len(chunk_files)} chunk files")
    
    total_rows = 0
    output_idx = 0
    buffer = []
    buffer_size = 0
    
    for chunk_file in tqdm(chunk_files, desc=f"{chunk_type}"):
        try:
            df = pd.read_parquet(os.path.join(CONFIG['temp_dir'], chunk_file))
            buffer.append(df)
            buffer_size += len(df)
            total_rows += len(df)
            
            # Write when buffer is full
            while buffer_size >= CONFIG['max_rows_per_file']:
                combined = pd.concat(buffer, ignore_index=True)
                
                # Take chunk to write
                to_write = combined.iloc[:CONFIG['max_rows_per_file']]
                remaining = combined.iloc[CONFIG['max_rows_per_file']:]
                
                # Write it
                out_path = os.path.join(CONFIG['output_dir'], f'{chunk_type}_part{output_idx:04d}.parquet')
                to_write.to_parquet(out_path, index=False, compression='snappy')
                output_idx += 1
                
                # Keep remainder
                buffer = [remaining] if len(remaining) > 0 else []
                buffer_size = len(remaining) if len(remaining) > 0 else 0
                
                del combined, to_write
                gc.collect()
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Write remaining
    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        out_path = os.path.join(CONFIG['output_dir'], f'{chunk_type}_part{output_idx:04d}.parquet')
        combined.to_parquet(out_path, index=False, compression='snappy')
        output_idx += 1
        del combined
        gc.collect()
    
    print(f"✓ {chunk_type}: {total_rows:,} rows in {output_idx} files")
    return total_rows


def create_small_balanced_sample(n_per_class: int = 500_000):
    """Create a smaller balanced sample without loading everything"""
    print(f"\n{'='*60}")
    print(f"CREATING BALANCED SAMPLE ({n_per_class:,} per class)")
    print(f"{'='*60}")
    
    human_files = sorted([f for f in os.listdir(CONFIG['output_dir']) if f.startswith('human_part')])
    ai_files = sorted([f for f in os.listdir(CONFIG['output_dir']) if f.startswith('ai_part')])
    
    # Sample from human
    print("Sampling human texts...")
    human_samples = []
    needed = n_per_class
    
    for f in human_files:
        if needed <= 0:
            break
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        take = min(len(df), needed)
        sample = df.sample(n=take, random_state=42) if take < len(df) else df
        human_samples.append(sample)
        needed -= take
        del df
        gc.collect()
    
    human_df = pd.concat(human_samples, ignore_index=True)
    print(f"  Got {len(human_df):,} human samples")
    del human_samples
    gc.collect()
    
    # Sample from AI
    print("Sampling AI texts...")
    ai_samples = []
    needed = n_per_class
    
    for f in ai_files:
        if needed <= 0:
            break
        df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
        take = min(len(df), needed)
        sample = df.sample(n=take, random_state=42) if take < len(df) else df
        ai_samples.append(sample)
        needed -= take
        del df
        gc.collect()
    
    ai_df = pd.concat(ai_samples, ignore_index=True)
    print(f"  Got {len(ai_df):,} AI samples")
    del ai_samples
    gc.collect()
    
    # Combine WITHOUT full shuffle - just interleave
    print("Interleaving (memory-efficient)...")
    
    # Reset indices
    human_df = human_df.reset_index(drop=True)
    ai_df = ai_df.reset_index(drop=True)
    
    # Add class indicator and combine
    human_df['_order'] = range(0, len(human_df) * 2, 2)  # Even positions
    ai_df['_order'] = range(1, len(ai_df) * 2, 2)        # Odd positions
    
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    del human_df, ai_df
    gc.collect()
    
    # Sort by interleave order (much cheaper than shuffle)
    combined = combined.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)
    
    # Save
    out_path = os.path.join(CONFIG['output_dir'], 'balanced_sample.parquet')
    combined.to_parquet(out_path, index=False, compression='snappy')
    print(f"✓ Saved {len(combined):,} texts to {out_path}")
    
    # Small CSV preview
    combined.head(10000).to_csv(os.path.join(CONFIG['output_dir'], 'preview.csv'), index=False)
    
    return len(combined)


def show_stats():
    """Show final statistics"""
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    
    human_count = 0
    ai_count = 0
    
    for f in os.listdir(CONFIG['output_dir']):
        if not f.endswith('.parquet') or 'sample' in f:
            continue
        try:
            df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
            if f.startswith('human'):
                human_count += len(df)
            elif f.startswith('ai'):
                ai_count += len(df)
            del df
        except:
            pass
    
    print(f"Human texts: {human_count:,}")
    print(f"AI texts: {ai_count:,}")
    print(f"Total: {human_count + ai_count:,}")
    
    # Disk size
    total_size = sum(
        os.path.getsize(os.path.join(CONFIG['output_dir'], f))
        for f in os.listdir(CONFIG['output_dir'])
        if f.endswith('.parquet')
    )
    print(f"Disk usage: {total_size/1e9:.2f} GB")


def main():
    print("="*60)
    print("ULTRA MEMORY-EFFICIENT MERGER")
    print("="*60)
    
    # Merge human chunks
    human_total = merge_chunks_simple('human')
    
    # Merge AI chunks  
    ai_total = merge_chunks_simple('ai')
    
    # Create balanced sample (500K per class = 1M total, very safe)
    if human_total > 0 and ai_total > 0:
        create_small_balanced_sample(n_per_class=500_000)
    
    show_stats()
    
    print("\n✓ COMPLETE!")


if __name__ == "__main__":
    main()