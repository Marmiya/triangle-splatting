"""
Parallel PLY file loader for large mesh triangle files
Optimized for loading millions of vertices efficiently
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Tuple
from plyfile import PlyData
from .triangle_model import BasicPointCloud


def load_ply_chunk(file_path: str, start_vertex: int, num_vertices: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a chunk of vertices from PLY file
    NOTE: This is conceptual - plyfile doesn't support partial loading natively
    """
    # This would require a custom PLY reader for true parallel loading
    # For now, we'll implement memory-efficient processing instead
    pass


def process_vertices_parallel(vertices, num_workers: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process vertex data in parallel chunks
    """
    num_vertices = len(vertices)
    chunk_size = num_vertices // num_workers
    
    print(f"Processing {num_vertices} vertices with {num_workers} workers, chunk_size={chunk_size}")
    
    def process_chunk(start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a chunk of vertices"""
        chunk_vertices = vertices[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        # Pre-allocate arrays for this chunk
        positions = np.empty((chunk_size, 3), dtype=np.float32)
        colors = np.empty((chunk_size, 3), dtype=np.float32)
        normals = np.empty((chunk_size, 3), dtype=np.float32)
        
        # Process chunk data
        positions[:, 0] = chunk_vertices['x']
        positions[:, 1] = chunk_vertices['y']
        positions[:, 2] = chunk_vertices['z']
        
        colors[:, 0] = chunk_vertices['red'] / 255.0
        colors[:, 1] = chunk_vertices['green'] / 255.0
        colors[:, 2] = chunk_vertices['blue'] / 255.0
        
        normals[:, 0] = chunk_vertices['nx']
        normals[:, 1] = chunk_vertices['ny']
        normals[:, 2] = chunk_vertices['nz']
        
        return positions, colors, normals
    
    # Create chunks
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_vertices)
        if start_idx < num_vertices:
            chunks.append((start_idx, end_idx))
    
    # Process chunks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, start, end): (start, end) 
            for start, end in chunks
        }
        
        for future in as_completed(future_to_chunk):
            start, end = future_to_chunk[future]
            try:
                chunk_positions, chunk_colors, chunk_normals = future.result()
                results.append((start, chunk_positions, chunk_colors, chunk_normals))
                print(f"Processed chunk [{start}:{end}]")
            except Exception as e:
                print(f"Error processing chunk [{start}:{end}]: {e}")
                raise
    
    # Sort results by start index and concatenate
    results.sort(key=lambda x: x[0])
    
    print("Concatenating parallel results...")
    positions = np.concatenate([r[1] for r in results], axis=0)
    colors = np.concatenate([r[2] for r in results], axis=0)
    normals = np.concatenate([r[3] for r in results], axis=0)
    
    return positions, colors, normals


def fetchPlyParallel(path: str, num_workers: int = 4) -> BasicPointCloud:
    """
    Load PLY file with parallel processing
    """
    print(f"Loading large PLY file with {num_workers} parallel workers: {path}")
    start_time = time.time()
    
    # Load PLY data (this is still single-threaded due to plyfile limitations)
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    num_vertices = len(vertices)
    
    load_time = time.time() - start_time
    print(f"PLY file loaded in {load_time:.2f}s, processing {num_vertices} vertices in parallel...")
    
    # Process data in parallel
    process_start = time.time()
    positions, colors, normals = process_vertices_parallel(vertices, num_workers)
    
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    
    print(f"Parallel processing completed in {process_time:.2f}s")
    print(f"Total PLY loading time: {total_time:.2f}s")
    print(f"Performance: {num_vertices / total_time:.0f} vertices/second")
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def split_ply_for_parallel_loading(input_path: str, output_dir: str, num_chunks: int = 4) -> List[str]:
    """
    Split large PLY file into smaller chunks for parallel loading
    This is a preprocessing step that can be done once
    """
    print(f"Splitting PLY file into {num_chunks} chunks...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load original PLY
    plydata = PlyData.read(input_path)
    vertices = plydata['vertex']
    num_vertices = len(vertices)
    chunk_size = num_vertices // num_chunks
    
    chunk_files = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_vertices) if i < num_chunks - 1 else num_vertices
        
        chunk_vertices = vertices[start_idx:end_idx]
        chunk_file = os.path.join(output_dir, f'mesh_triangles_chunk_{i:02d}.ply')
        
        # Save chunk as PLY
        from plyfile import PlyElement
        vertex_element = PlyElement.describe(chunk_vertices, 'vertex')
        chunk_plydata = PlyData([vertex_element])
        chunk_plydata.write(chunk_file)
        
        chunk_files.append(chunk_file)
        print(f"Saved chunk {i+1}/{num_chunks}: {chunk_file} ({len(chunk_vertices)} vertices)")
    
    return chunk_files


def load_ply_chunks_parallel(chunk_files: List[str], num_workers: int = None) -> BasicPointCloud:
    """
    Load multiple PLY chunk files in parallel and combine them
    """
    if num_workers is None:
        num_workers = min(len(chunk_files), 4)
    
    print(f"Loading {len(chunk_files)} PLY chunks with {num_workers} workers...")
    start_time = time.time()
    
    def load_single_chunk(chunk_file: str) -> BasicPointCloud:
        """Load a single PLY chunk file"""
        plydata = PlyData.read(chunk_file)
        vertices = plydata['vertex']
        num_vertices = len(vertices)
        
        positions = np.empty((num_vertices, 3), dtype=np.float32)
        colors = np.empty((num_vertices, 3), dtype=np.float32)
        normals = np.empty((num_vertices, 3), dtype=np.float32)
        
        positions[:, 0] = vertices['x']
        positions[:, 1] = vertices['y']
        positions[:, 2] = vertices['z']
        
        colors[:, 0] = vertices['red'] / 255.0
        colors[:, 1] = vertices['green'] / 255.0
        colors[:, 2] = vertices['blue'] / 255.0
        
        normals[:, 0] = vertices['nx']
        normals[:, 1] = vertices['ny']
        normals[:, 2] = vertices['nz']
        
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    # Load chunks in parallel
    chunk_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(load_single_chunk, chunk_file): chunk_file 
            for chunk_file in chunk_files
        }
        
        for future in as_completed(future_to_file):
            chunk_file = future_to_file[future]
            try:
                chunk_pcd = future.result()
                chunk_results.append((chunk_file, chunk_pcd))
                print(f"Loaded chunk: {os.path.basename(chunk_file)} ({len(chunk_pcd.points)} vertices)")
            except Exception as e:
                print(f"Error loading chunk {chunk_file}: {e}")
                raise
    
    # Sort by filename and combine
    chunk_results.sort(key=lambda x: x[0])
    
    print("Combining chunk results...")
    all_positions = np.concatenate([result[1].points for result in chunk_results], axis=0)
    all_colors = np.concatenate([result[1].colors for result in chunk_results], axis=0)
    all_normals = np.concatenate([result[1].normals for result in chunk_results], axis=0)
    
    total_time = time.time() - start_time
    total_vertices = len(all_positions)
    
    print(f"Parallel chunk loading completed in {total_time:.2f}s")
    print(f"Total vertices loaded: {total_vertices}")
    print(f"Performance: {total_vertices / total_time:.0f} vertices/second")
    
    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)