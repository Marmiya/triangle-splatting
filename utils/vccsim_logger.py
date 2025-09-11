#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCCSim Training Logger

A specialized logging utility for VCCSim Triangle Splatting training that outputs
progress in format expected by the C++ VCCSim panel manager for real-time monitoring.

This logger provides:
- Progress logging compatible with C++ progress parsing
- Triangle statistics tracking  
- PSNR caching for efficient progress updates
- Configurable logging frequencies
- File and console output with timestamps
"""

import os
from datetime import datetime


class VCCSimTrainingLogger:
    """Logger that outputs progress in format expected by VCCSim C++ manager"""
    
    def __init__(self, log_file_path):
        # Normalize path to use forward slashes and resolve relative paths
        self.log_file_path = os.path.normpath(os.path.abspath(log_file_path)).replace('\\', '/')
        self.ensure_log_dir()
        
        # Track latest PSNR for progress logging
        self.latest_train_psnr = None
        self.latest_test_psnr = None
        
    def ensure_log_dir(self):
        """Ensure log directory exists"""
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
    def log(self, message, iteration=None, loss=None, psnr_val=None):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format for C++ progress parsing
        if iteration is not None:
            if loss is not None:
                log_msg = f"[{timestamp}] Iteration {iteration}: Loss={loss:.6f}"
                if psnr_val is not None:
                    log_msg += f", PSNR={psnr_val:.2f}"
            else:
                log_msg = f"[{timestamp}] Iteration {iteration}: {message}"
        else:
            log_msg = f"[{timestamp}] {message}"
            
        print(log_msg)  # Console output
        
        # Write to log file for C++ monitoring
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_training_progress(self, iteration, max_iterations):
        """Log basic training progress for C++ progress bar parsing"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_msg = f"[{timestamp}] Iteration {iteration}/{max_iterations}"
        
        print(progress_msg)  # Console output
        
        # Write to log file for C++ monitoring
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(progress_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_loss_update(self, iteration, loss, speed_iter_per_sec=None):
        """Log loss value updates with optional speed information"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loss_msg = f"[{timestamp}] [LOSS] Iteration {iteration}: Loss={loss:.6f}"
        
        if speed_iter_per_sec is not None and speed_iter_per_sec > 0:
            loss_msg += f", {speed_iter_per_sec:.1f} iter/s"
        
        print(loss_msg)  # Console output
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(loss_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_psnr_update(self, iteration, psnr_val):
        """Log PSNR value updates"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        psnr_msg = f"[{timestamp}] [PSNR] Iteration {iteration}: PSNR={psnr_val:.2f}"
        
        print(psnr_msg)  # Console output
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(psnr_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_triangle_count_update(self, iteration, triangle_count):
        """Log triangle count updates"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        triangle_msg = f"[{timestamp}] [TRIANGLES] Iteration {iteration}: Triangles={triangle_count}"
        
        print(triangle_msg)  # Console output
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(triangle_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_triangle_stats(self, iteration, triangle_count):
        """Log detailed triangle statistics for analysis
        
        This provides triangle count information for detailed triangle evolution tracking.
        Speed information is now included in the frequent loss updates.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        stats_msg = f"[{timestamp}] [TRIANGLE STATS] Iteration {iteration}: {triangle_count} triangles"
            
        print(stats_msg)  # Console output
        
        # Write to log file for C++ monitoring  
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(stats_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
            
    def update_psnr_cache(self, train_psnr=None, test_psnr=None):
        """Update cached PSNR values for progress logging"""
        if train_psnr is not None:
            self.latest_train_psnr = train_psnr
        if test_psnr is not None:
            self.latest_test_psnr = test_psnr
    
    def log_training_metrics(self, iteration, loss, triangle_count, speed_iter_per_sec=None, 
                           train_psnr=None, test_psnr=None):
        """Log comprehensive training metrics in a single call"""
        # Build comprehensive stats message (timestamp will be added by self.log())
        stats_msg = f"Training Progress - Loss: {loss:.6f}, Triangles: {triangle_count}"
        
        if speed_iter_per_sec is not None and speed_iter_per_sec > 0:
            stats_msg += f", Speed: {speed_iter_per_sec:.1f} iter/s"
        if train_psnr is not None:
            stats_msg += f", Train PSNR: {train_psnr:.2f}"
        if test_psnr is not None:
            stats_msg += f", Test PSNR: {test_psnr:.2f}"
            
        self.log(stats_msg, iteration)
    
    def log_evaluation_results(self, iteration, config_name, pixel_loss, psnr_val, ssim_val, lpips_val=None):
        """Log evaluation results from validation runs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        eval_msg = f"[{timestamp}] [ITER {iteration}] Evaluating {config_name}: L1 {pixel_loss} PSNR {psnr_val} SSIM {ssim_val}"
        if lpips_val is not None:
            eval_msg += f" LPIPS {lpips_val}"
            
        print(eval_msg)  # Console output
        
        # Write to log file
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(eval_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def log_densification_stats(self, iteration, dead_count, triangle_count_before, triangle_count_after):
        """Log triangle densification statistics"""
        self.log(f"[TRIANGLE STATS] Densification at iteration {iteration}: {dead_count} dead, {triangle_count_before} -> {triangle_count_after} triangles")
    
    def log_pruning_stats(self, iteration, removed_count, triangle_count_before, triangle_count_after):
        """Log triangle pruning statistics"""  
        self.log(f"[TRIANGLE STATS] Final Pruning at iteration {iteration}: Removed {removed_count}, {triangle_count_before} -> {triangle_count_after} triangles")
    
    def log_initial_stats(self, triangle_count):
        """Log initial training statistics"""
        self.log(f"[TRIANGLE STATS] Initial triangles: {triangle_count}")
    
    def log_final_stats(self, final_triangle_count, initial_triangle_count):
        """Log final training statistics"""
        self.log(f"[TRIANGLE STATS] Final triangles: {final_triangle_count} (started with {initial_triangle_count})")
    
    def log_checkpoint_save(self, iteration):
        """Log checkpoint saving"""
        self.log(f"Saving Triangles", iteration)
    
    def log_debug_render_save(self, render_path):
        """Log debug render file saving"""
        self.log(f"[DEBUG] Saved render to: {render_path}")
    
    def log_completion(self):
        """Log training completion"""
        self.log("Training is done")
        self.log("Training complete.")


# ============================================================================
# LOGGING CONFIGURATION HELPER
# ============================================================================

class VCCSimLoggingConfig:
    """Configuration class for VCCSim logging frequencies and settings"""
    
    def __init__(self, 
                 loss_every_n=10,
                 psnr_every_n=200,
                 triangle_stats_every_n=100, 
                 enable_frequent_psnr=False,
                 frequent_psnr_interval=500):
        """
        Initialize logging configuration with individual frequencies for each metric
        
        Args:
            loss_every_n (int): Loss value updates every N iterations (includes speed)
            psnr_every_n (int): PSNR value updates every N iterations  
            triangle_stats_every_n (int): Triangle statistics every N iterations
            enable_frequent_psnr (bool): Enable frequent PSNR calculation (impacts performance)
            frequent_psnr_interval (int): PSNR calculation interval if frequent PSNR enabled
        """
        self.loss_every_n = loss_every_n
        self.psnr_every_n = psnr_every_n
        self.triangle_stats_every_n = triangle_stats_every_n
        self.enable_frequent_psnr = enable_frequent_psnr
        self.frequent_psnr_interval = frequent_psnr_interval
    
    @classmethod
    def default_performance_optimized(cls):
        """Default configuration optimized for training performance"""
        return cls(
            loss_every_n=10,           # Loss updates every 10 iterations (includes speed)
            psnr_every_n=200,          # PSNR updates every 200 iterations (less frequent)
            triangle_stats_every_n=100, # Triangle statistics every 100 iterations
            enable_frequent_psnr=False,
            frequent_psnr_interval=500
        )
    
    @classmethod
    def high_frequency_logging(cls):
        """High-frequency logging configuration (may impact performance)"""
        return cls(
            loss_every_n=5,            # More frequent loss updates (includes speed)
            psnr_every_n=100,          # More frequent PSNR updates
            triangle_stats_every_n=50, # More frequent triangle statistics
            enable_frequent_psnr=True,
            frequent_psnr_interval=250
        )
    
    @classmethod
    def minimal_logging(cls):
        """Minimal logging for maximum performance"""
        return cls(
            loss_every_n=50,           # Infrequent loss updates (includes speed)
            psnr_every_n=1000,         # Very infrequent PSNR updates
            triangle_stats_every_n=500, # Very infrequent triangle statistics
            enable_frequent_psnr=False,
            frequent_psnr_interval=1000
        )
    
    def should_log_loss(self, iteration):
        """Check if should log loss at this iteration"""
        return iteration % self.loss_every_n == 0
    
    def should_log_psnr(self, iteration):
        """Check if should log PSNR at this iteration"""
        return iteration % self.psnr_every_n == 0
    
    def should_log_triangle_stats(self, iteration):
        """Check if should log triangle statistics at this iteration"""
        return iteration % self.triangle_stats_every_n == 0
    
    def should_calculate_frequent_psnr(self, iteration):
        """Check if should calculate frequent PSNR at this iteration"""
        return self.enable_frequent_psnr and iteration % self.frequent_psnr_interval == 0 and iteration > 0