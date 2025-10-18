"""Timing tracker for evaluation pipeline stages."""
import time
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any


class TimingTracker:
    """Track and record execution times for different pipeline stages."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.timings: Dict[str, Any] = {
            'experiment': experiment_name,
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        self._current_stage: Optional[str] = None
        self._stage_start: Optional[float] = None

    @contextmanager
    def stage(self, stage_name: str):
        """Context manager to track a pipeline stage."""
        self._current_stage = stage_name
        self._stage_start = time.time()
        stage_start_time = datetime.now().isoformat()

        print(f"\n{'='*60}")
        print(f"[TIMING] Starting stage: {stage_name}")
        print(f"[TIMING] Stage start time: {stage_start_time}")
        print(f"{'='*60}\n")

        try:
            yield
        finally:
            elapsed = time.time() - self._stage_start
            stage_end_time = datetime.now().isoformat()

            self.timings['stages'][stage_name] = {
                'start_time': stage_start_time,
                'end_time': stage_end_time,
                'duration_seconds': elapsed,
                'duration_formatted': self._format_duration(elapsed)
            }

            print(f"\n{'='*60}")
            print(f"[TIMING] Completed stage: {stage_name}")
            print(f"[TIMING] Duration: {self._format_duration(elapsed)}")
            print(f"[TIMING] Stage end time: {stage_end_time}")
            print(f"{'='*60}\n")

            self._current_stage = None
            self._stage_start = None

    def record_metadata(self, **kwargs):
        """Record additional metadata about the experiment."""
        if 'metadata' not in self.timings:
            self.timings['metadata'] = {}
        self.timings['metadata'].update(kwargs)

    def save(self):
        """Save timing data to JSON file."""
        self.timings['end_time'] = datetime.now().isoformat()

        # Calculate total duration
        if self.timings['stages']:
            total_seconds = sum(
                stage['duration_seconds']
                for stage in self.timings['stages'].values()
            )
            self.timings['total_duration_seconds'] = total_seconds
            self.timings['total_duration_formatted'] = self._format_duration(total_seconds)

        # Save to file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timing_file = self.output_dir / f"timing_{self.experiment_name}.json"

        with open(timing_file, 'w') as f:
            json.dump(self.timings, f, indent=2)

        print(f"\n{'='*60}")
        print(f"[TIMING] Timing data saved to: {timing_file}")
        print(f"[TIMING] Total experiment duration: {self.timings.get('total_duration_formatted', 'N/A')}")
        print(f"{'='*60}\n")

        # Print summary
        self._print_summary()

        return timing_file

    def _print_summary(self):
        """Print timing summary."""
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Start: {self.timings['start_time']}")
        print(f"End: {self.timings['end_time']}")
        print(f"Total Duration: {self.timings.get('total_duration_formatted', 'N/A')}")
        print("\nStage Breakdown:")
        print("-" * 60)

        for stage_name, stage_data in self.timings['stages'].items():
            duration = stage_data['duration_formatted']
            print(f"  {stage_name:20s}: {duration}")

        print("="*60 + "\n")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        else:
            return f"{secs:.2f}s"


def compare_timings(timing_files: list[str]):
    """Compare timing data from multiple experiments.

    Returns:
        List of dictionaries with timing data from each experiment.
    """
    data = []
    for timing_file in timing_files:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)

        row = {
            'experiment': timing_data['experiment'],
            'start_time': timing_data['start_time'],
            'end_time': timing_data['end_time'],
            'total_duration_seconds': timing_data.get('total_duration_seconds', 0),
            'total_duration_formatted': timing_data.get('total_duration_formatted', 'N/A')
        }

        # Add stage durations
        for stage_name, stage_data in timing_data.get('stages', {}).items():
            row[f'{stage_name}_seconds'] = stage_data['duration_seconds']
            row[f'{stage_name}_formatted'] = stage_data['duration_formatted']

        # Add metadata if present
        if 'metadata' in timing_data:
            row['metadata'] = timing_data['metadata']

        data.append(row)

    return data
