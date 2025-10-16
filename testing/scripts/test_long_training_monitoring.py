#!/usr/bin/env python3
"""
Comprehensive training monitoring script for long-term DMC training.
Supports 100+ epochs with CPU/GPU utilization monitoring and detailed metrics logging.
Integrated with Guandan-specific metrics analysis.
"""

import os
import sys
import time
import psutil
import subprocess
import json
import threading
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import ray
from guandan.rllib.trainers import create_dmc_trainer

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_info.append({
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    })
            return gpu_info
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
    return []

def get_cpu_utilization():
    """Get CPU utilization"""
    try:
        return psutil.cpu_percent(interval=1)
    except Exception as e:
        print(f"Warning: Could not get CPU info: {e}")
        return 0

def get_memory_utilization():
    """Get memory utilization"""
    try:
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
    except Exception as e:
        print(f"Warning: Could not get memory info: {e}")
        return {'percent': 0, 'used_gb': 0, 'total_gb': 0}

def get_disk_utilization():
    """Get disk utilization"""
    try:
        disk = psutil.disk_usage('/')
        return {
            'percent': disk.percent,
            'used_gb': disk.used / (1024**3),
            'total_gb': disk.total / (1024**3)
        }
    except Exception as e:
        print(f"Warning: Could not get disk info: {e}")
        return {'percent': 0, 'used_gb': 0, 'total_gb': 0}

def log_system_metrics(epoch, phase="training"):
    """Log system metrics for current epoch"""
    cpu_util = get_cpu_utilization()
    memory_info = get_memory_utilization()
    disk_info = get_disk_utilization()
    gpu_info = get_gpu_utilization()
    
    print(f"  📊 System Metrics ({phase}):")
    print(f"    CPU: {cpu_util:.1f}%")
    print(f"    Memory: {memory_info['percent']:.1f}% ({memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB)")
    print(f"    Disk: {disk_info['percent']:.1f}% ({disk_info['used_gb']:.1f}GB/{disk_info['total_gb']:.1f}GB)")
    
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"    GPU {i}: {gpu['utilization']}% | Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB | Temp: {gpu['temperature']}°C")
    
    return {
        'cpu_utilization': cpu_util,
        'memory_percent': memory_info['percent'],
        'memory_used_gb': memory_info['used_gb'],
        'disk_percent': disk_info['percent'],
        'disk_used_gb': disk_info['used_gb'],
        'gpu_info': gpu_info
    }

def save_checkpoint(trainer, epoch, results_dir):
    """Save training checkpoint"""
    checkpoint_dir = os.path.join(results_dir, f"checkpoint_epoch_{epoch}")
    try:
        checkpoint_path = trainer.save(checkpoint_dir)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"  ⚠️  Could not save checkpoint: {e}")
        return None

def start_ray_monitor(results_dir):
    """Start Ray results monitor in a separate thread"""
    monitor_process = None
    
    def monitor_thread():
        nonlocal monitor_process
        try:
            # Wait for Ray to initialize and create result files
            time.sleep(30)
            
            # Import and run the monitor
            script_dir = os.path.dirname(os.path.abspath(__file__))
            monitor_script = os.path.join(script_dir, "monitor_ray_training.py")
            
            if os.path.exists(monitor_script):
                # Use Popen so it doesn't block the thread
                monitor_process = subprocess.Popen([
                    sys.executable, 
                    monitor_script,
                    "--output-dir", results_dir,
                    "--interval", "15"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️  Monitor thread error: {e}")
    
    thread = threading.Thread(target=monitor_thread, daemon=True)
    thread.start()
    
    # Return both thread and a function to stop the monitor
    def stop_monitor():
        if monitor_process and monitor_process.poll() is None:
            try:
                monitor_process.terminate()
                monitor_process.wait(timeout=5)
            except:
                monitor_process.kill()
    
    return thread, stop_monitor

def main():
    """Main training function with comprehensive monitoring"""
    print("=" * 80)
    print("🚀 DMC Long-Term Training with Comprehensive Monitoring")
    print("=" * 80)
    
    # Configuration
    # Allow external override via environment variables for orchestrated runs
    try:
        total_epochs = int(os.environ.get("DANZERO_TOTAL_EPOCHS", "3"))  # default 3 for quick test
    except ValueError:
        total_epochs = 3
    checkpoint_freq = 100  # Save checkpoint every 10 epochs
    metrics_save_freq = 2  # Save metrics every 2 epochs
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Place all training results under project results/ folder
    prefix = os.environ.get("DANZERO_RESULTS_PREFIX", "training_results")
    results_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "results")
    results_root = os.path.abspath(results_root)
    os.makedirs(results_root, exist_ok=True)
    results_dir = os.path.join(results_root, f"{prefix}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"🎯 Training for {total_epochs} epochs")
    print(f"💾 Checkpoints every {checkpoint_freq} epochs")
    print(f"📊 Metrics saved every {metrics_save_freq} epochs")
    print()
    
    # Start integrated Ray monitor (optional via env)
    if os.environ.get("DANZERO_DISABLE_MONITOR") == "1":
        stop_monitor = lambda: None
    else:
        print("🔍 Starting integrated Ray monitor...")
        monitor_thread, stop_monitor = start_ray_monitor(results_dir)
        print()
    
    # Initialize Ray with optimized settings
    print("🔧 Initializing Ray...")
    ray.shutdown()
    ray.init(
        num_cpus=64,
        num_gpus=2,
        object_store_memory=10 * 1024 * 1024 * 1024,  # 10GB
        ignore_reinit_error=True
    )
    
    try:
        # Create DMC trainer with optimized settings
        print("🎯 Creating DMC trainer...")
        trainer = create_dmc_trainer(
            env_config={
                "observation_mode": "comprehensive",
                "use_internal_adapters": False,
                "max_steps": 1000,
            },
            num_workers=50,
            num_envs_per_worker=4,
            num_gpus=1.0,
            num_gpus_per_worker=0.0,
            lr=1e-3,
            batch_size=200,
            epsilon_start=0.2,
            epsilon_end=0.05,
            epsilon_decay_steps=10000,
        )
        
        print("✅ DMC trainer created successfully!")
        print()
        
        # Training variables
        cumulative_episodes = 0
        cumulative_steps = 0
        total_loss = 0.0
        best_loss = float('inf')
        
        # Initialize metrics log
        metrics_log = []
        training_start_time = time.time()
        
        print("🏃 Starting training...")
        print("=" * 80)
        
        # Training loop with comprehensive monitoring
        for epoch in range(total_epochs):
            epoch_start_time = time.time()
            
            print(f"\n📈 [Epoch {epoch+1}/{total_epochs}] Starting...")
            
            # Get system metrics before training
            pre_metrics = log_system_metrics(epoch+1, "pre-training")
            
            # Run training iteration
            try:
                result = trainer.train()
                iter_time = time.time() - epoch_start_time
                
                # Get system metrics after training
                post_metrics = log_system_metrics(epoch+1, "post-training")
                
                # Extract training metrics
                episodes_this_iter = 0
                timesteps_total = 0
                episode_reward_mean = 0.0
                loss = 0.0
                
                try:
                    # Handle both dict and list results
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                        result_dict = result[0].copy()
                    elif isinstance(result, dict):
                        result_dict = result.copy()
                    else:
                        result_dict = {}
                    
                    # Debug output removed - episode counting fixed
                    
                    # Extract stats values (handle Stats objects)
                    def extract_stat(stat_obj):
                        if stat_obj is None:
                            return 0
                        if hasattr(stat_obj, 'peek'):
                            return stat_obj.peek()
                        if hasattr(stat_obj, 'value'):
                            return stat_obj.value
                        return stat_obj
                    
                    # RLlib new API stack stores metrics under specific keys
                    # Check for learner results
                    if isinstance(result_dict, dict) and 'learner_results' in result_dict:
                        learner_results = result_dict['learner_results']
                        
                        # Ensure learner_results is a dict (not a list)
                        if isinstance(learner_results, list) and len(learner_results) > 0:
                            learner_results = learner_results[0]
                        
                        if isinstance(learner_results, dict):
                            # Extract __all_modules__ metrics
                            if '__all_modules__' in learner_results:
                                all_modules = learner_results['__all_modules__']
                                
                                if isinstance(all_modules, dict):
                                    timesteps_total = extract_stat(all_modules.get('num_env_steps_trained', 0))
                            
                            # Get loss from first agent
                            if 'agent_0' in learner_results:
                                agent_0 = learner_results['agent_0']
                                if isinstance(agent_0, dict):
                                    loss_stat = agent_0.get('loss')
                                    loss = extract_stat(loss_stat)
                    
                    # Also check 'learners' key (alternative location for learner metrics)
                    if 'learners' in result_dict:
                        learners = result_dict['learners']
                        if isinstance(learners, dict):
                            # Try to find loss in any learner
                            for learner_key, learner_data in learners.items():
                                if isinstance(learner_data, dict):
                                    # Check for loss in this learner
                                    if 'loss' in learner_data:
                                        loss = extract_stat(learner_data['loss'])
                                        break
                                    # Check for agent-specific losses
                                    for agent_key in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
                                        if agent_key in learner_data:
                                            agent_data = learner_data[agent_key]
                                            if isinstance(agent_data, dict) and 'loss' in agent_data:
                                                loss = extract_stat(agent_data['loss'])
                                                break
                                    if loss > 0:
                                        break
                    
                    # Check env_runners for episode data (new API stack)
                    if 'env_runners' in result_dict:
                        env_runners = result_dict['env_runners']
                        if isinstance(env_runners, dict):
                            episodes_this_iter = int(extract_stat(env_runners.get('num_episodes', 0)))
                            timesteps_total = int(extract_stat(env_runners.get('num_env_steps_sampled', 0)))
                    
                    # Also check env_runner_results for episode data (fallback)
                    if 'env_runner_results' in result_dict:
                        env_results = result_dict['env_runner_results']
                        episodes_this_iter = int(extract_stat(env_results.get('num_episodes', 0)))
                    
                    # Fallback: Try to extract from top-level keys (older RLlib format)
                    if timesteps_total == 0 and 'num_env_steps_sampled' in result_dict:
                        timesteps_total = int(extract_stat(result_dict.get('num_env_steps_sampled', 0)))
                    
                    if episodes_this_iter == 0 and 'episodes_this_iter' in result_dict:
                        episodes_this_iter = int(extract_stat(result_dict.get('episodes_this_iter', 0)))
                    
                    cumulative_episodes += episodes_this_iter
                    cumulative_steps += timesteps_total
                    total_loss += loss
                    
                    # Track best loss
                    if loss > 0 and loss < best_loss:
                        best_loss = loss
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not extract all metrics: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Log epoch results
                print(f"📊 [Epoch {epoch+1}/{total_epochs}] "
                      f"{iter_time:.2f}s | "
                      f"Episodes: {cumulative_episodes:4d} (+{episodes_this_iter}) | "
                      f"Steps: {cumulative_steps:9d} | "
                      f"Loss: {loss:.6f} | "
                      f"Avg Loss: {total_loss/(epoch+1):.6f} | "
                      f"Best Loss: {best_loss:.6f}")
                
                # Store metrics for this epoch
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'timestamp': datetime.now().isoformat(),
                    'time_seconds': iter_time,
                    'episodes_total': cumulative_episodes,
                    'episodes_this_iter': episodes_this_iter,
                    'steps_total': cumulative_steps,
                    'steps_this_iter': timesteps_total,
                    'loss': loss,
                    'avg_loss': total_loss / (epoch + 1),
                    'best_loss': best_loss,
                    'pre_training_metrics': pre_metrics,
                    'post_training_metrics': post_metrics
                }
                metrics_log.append(epoch_metrics)
                
                # Save metrics periodically
                if (epoch + 1) % metrics_save_freq == 0:
                    metrics_file = os.path.join(results_dir, f"metrics_epoch_{epoch+1}.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics_log, f, indent=2)
                    print(f"  📊 Metrics saved to {metrics_file}")
                
                # Save checkpoint periodically
                if (epoch + 1) % checkpoint_freq == 0:
                    checkpoint_path = save_checkpoint(trainer, epoch+1, results_dir)
                
                # Progress summary every 20 epochs
                if (epoch + 1) % 20 == 0:
                    elapsed_time = time.time() - training_start_time
                    avg_time_per_epoch = elapsed_time / (epoch + 1)
                    estimated_total_time = avg_time_per_epoch * total_epochs
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"\n📈 Progress Summary (Epoch {epoch+1}):")
                    print(f"  ⏱️  Elapsed: {elapsed_time/3600:.1f}h | Remaining: {remaining_time/3600:.1f}h")
                    print(f"  📊 Episodes: {cumulative_episodes} | Steps: {cumulative_steps}")
                    print(f"  🎯 Loss: {loss:.6f} | Best: {best_loss:.6f}")
                    print(f"  💾 Checkpoints: {len([f for f in os.listdir(results_dir) if f.startswith('checkpoint_')])}")
                
            except Exception as e:
                print(f"\n❌ Error during epoch {epoch+1}: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("\nFull traceback:")
                traceback.print_exc()
                print("\n")
                continue
        
        # Final summary
        total_time = time.time() - training_start_time
        print("\n" + "=" * 80)
        print("🎉 Training completed!")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"📊 Total epochs: {total_epochs}")
        print(f"🎮 Total episodes: {cumulative_episodes}")
        print(f"👣 Total steps: {cumulative_steps}")
        print(f"📈 Final loss: {total_loss/total_epochs:.6f}")
        print(f"🏆 Best loss: {best_loss:.6f}")
        print(f"📁 Results saved in: {results_dir}")
        
        # Save final metrics and summary
        final_metrics_file = os.path.join(results_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        
        # Create training summary
        summary = {
            'training_completed': True,
            'total_epochs': total_epochs,
            'total_time_hours': total_time / 3600,
            'total_episodes': cumulative_episodes,
            'total_steps': cumulative_steps,
            'final_loss': total_loss / total_epochs,
            'best_loss': best_loss,
            'results_directory': results_dir,
            'checkpoints_saved': len([f for f in os.listdir(results_dir) if f.startswith('checkpoint_')]),
            'completion_time': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(results_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Final metrics saved to {final_metrics_file}")
        print(f"📋 Training summary saved to {summary_file}")
        print("=" * 80)
        
        # Run Guandan-specific analysis
        print("\n📊 Running Guandan-specific metrics analysis...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            analyze_script = os.path.join(script_dir, "analyze_guandan_metrics.py")
            
            if os.path.exists(analyze_script):
                analysis_output = os.path.join(results_dir, "guandan_analysis.json")
                # Prefer passing the actual Ray logdir from the trainer to ensure plots are generated
                ray_dir = getattr(trainer, "logdir", None)
                cmd = [
                    sys.executable,
                    analyze_script,
                ]
                if isinstance(ray_dir, str) and os.path.isdir(ray_dir):
                    cmd += ["--ray-dir", ray_dir]
                cmd += ["--output", analysis_output]
                subprocess.run(cmd, check=False, timeout=60)  # 60 second timeout
                print(f"✅ Guandan analysis saved to {analysis_output}")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Guandan analysis timed out after 60 seconds")
        except Exception as e:
            print(f"⚠️  Could not run Guandan analysis: {e}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            # Stop monitor process first
            print("  🛑 Stopping monitor process...")
            stop_monitor()
        except:
            pass
        
        try:
            print("  🛑 Stopping trainer...")
            trainer.stop()
        except:
            pass
        
        try:
            print("  🛑 Shutting down Ray...")
            ray.shutdown()
        except:
            pass
        
        print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()
