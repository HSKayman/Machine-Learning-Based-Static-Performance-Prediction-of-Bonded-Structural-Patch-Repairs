"""
Orchestrator for the ANN experiment pipeline.
Runs all parts of the pipeline: preprocess -> synthetic -> train -> visualize
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import yaml

# Orchestrates the full ANN training pipeline
class PipelineOrchestrator:

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.start_time = None
        self.log_lines = []
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log a message with timestamp.
    def log(self, message: str, level: str = 'INFO') -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        self.log_lines.append(log_msg)
    
    def runner(self, stage_name: str, script: str, args: list) -> bool:
        self.log(f"Starting stage: {stage_name}")
        stage_start = time.time()
        
        cmd = [sys.executable, script] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                cwd=Path(__file__).parent  # Run from pipeline directory
            )
            
            elapsed = time.time() - stage_start
            
            if result.returncode == 0:
                self.log(f"Completed {stage_name} in {elapsed:.1f}s")
                return True
            else:
                self.log(f"Stage {stage_name} failed with return code {result.returncode}", 'ERROR')
                return False
                
        except Exception as e:
            self.log(f"Stage {stage_name} failed with exception: {e}", 'ERROR')
            return False
    
    def run(self, dataset: str = 'both', 
            skip_synthetic: bool = False,
            skip_training: bool = False,
            skip_visualization: bool = False) -> bool:
        """
        Run the full pipeline.
        
        Args:
            dataset: 'patched', 'unpatched', or 'both'
            skip_synthetic: Skip synthetic data generation
            skip_training: Skip training (use existing results)
            skip_visualization: Skip visualization generation
            
        Returns:
            True if all stages succeeded
        """
        self.start_time = time.time()
        self.log("="*60)
        self.log("ANN EXPERIMENT PIPELINE")
        self.log("="*60)
        self.log(f"Dataset: {dataset}")
        self.log(f"Config: {self.config_path}")
        self.log(f"Output: {self.output_dir}")
        
        common_args = ['--config', self.config_path, '--dataset', dataset]
        success = True
        
        # Stage 1: Preprocessing
        self.log("-"*60)
        if not self.runner(
            "Preprocessing",
            "preprocess.py",
            common_args
        ):
            self.log("Pipeline failed at preprocessing stage", 'ERROR')
            return False
        
        # Stage 2: Synthetic data generation
        if not skip_synthetic and self.config.get('synthetic', {}).get('enabled', True):
            self.log("-"*60)
            if not self.runner(
                "Synthetic Data Generation",
                "synthetic.py",
                common_args
            ):
                self.log("Pipeline failed at synthetic data generation", 'ERROR')
                return False
        else:
            self.log("Skipping synthetic data generation")
        
        # Stage 3: Training
        if not skip_training:
            self.log("-"*60)
            train_args = common_args + ['--use-augmented']
            if not self.runner(
                "Training & Grid Search",
                "train.py",
                train_args
            ):
                self.log("Pipeline failed at training stage", 'ERROR')
                return False
        else:
            self.log("Skipping training (using existing results)")
        
        # Stage 4: Visualization
        if not skip_visualization:
            self.log("-"*60)
            if not self.runner(
                "Visualization",
                "visualize.py",
                common_args
            ):
                self.log("Pipeline failed at visualization stage", 'ERROR')
                success = False
        else:
            self.log("Skipping visualization")
        
        # Summary
        total_time = time.time() - self.start_time
        self.log("-"*60)
        self.log(f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        self.log("="*60)
        
        # List outputs
        self.log("\nGenerated files:")
        for f in sorted(self.output_dir.glob('*')):
            self.log(f"  - {f.name}")
        
        # Save log
        log_path = self.output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.log_lines))
        self.log(f"\nLog saved to {log_path}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description='Run the ANN experiment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run full pipeline on both datasets
        python orchestrator.py
        
        # Run only on patched dataset
        python orchestrator.py --dataset patched
        
        # Skip synthetic data generation
        python orchestrator.py --no-synthetic
        
        # Only regenerate visualizations
        python orchestrator.py --skip-training
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['patched', 'unpatched', 'both'],
                        default='both')
    parser.add_argument('--no-synthetic', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--skip-viz', action='store_true')
    
    args = parser.parse_args()
    
    # Change to script directory for relative path resolution
    os.chdir(Path(__file__).parent)
    
    orchestrator = PipelineOrchestrator(args.config)
    
    success = orchestrator.run(
        dataset=args.dataset,
        skip_synthetic=args.no_synthetic,
        skip_training=args.skip_training,
        skip_visualization=args.skip_viz
    )

main()

