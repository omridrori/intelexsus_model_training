import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def plot_comparison(our_model_log_path, baseline_model_log_path, from_scratch_model_log_path, output_image_path):
    """
    Plots the accuracy comparison between three models based on their training log histories.

    Args:
        our_model_log_path (Path): Path to the log_history.json for our model.
        baseline_model_log_path (Path): Path to the log_history.json for the baseline model.
        from_scratch_model_log_path (Path): Path to the log_history.json for the from-scratch model.
        output_image_path (Path): Path to save the output plot image.
    """
    with open(our_model_log_path, 'r') as f:
        our_model_logs = json.load(f)

    with open(baseline_model_log_path, 'r') as f:
        baseline_model_logs = json.load(f)

    with open(from_scratch_model_log_path, 'r') as f:
        from_scratch_model_logs = json.load(f)

    def extract_eval_data(logs):
        steps = []
        accuracies = []
        for entry in logs:
            if 'eval_accuracy' in entry:
                steps.append(entry['step'])
                accuracies.append(entry['eval_accuracy'])
        return steps, accuracies

    our_steps, our_accuracies = extract_eval_data(our_model_logs)
    baseline_steps, baseline_accuracies = extract_eval_data(baseline_model_logs)
    from_scratch_steps, from_scratch_accuracies = extract_eval_data(from_scratch_model_logs)

    plt.figure(figsize=(12, 8))
    plt.plot(our_steps, our_accuracies, marker='o', linestyle='-', label='Continual Pretrained mBERT')
    plt.plot(baseline_steps, baseline_accuracies, marker='x', linestyle='--', label='Baseline mBERT')
    plt.plot(from_scratch_steps, from_scratch_accuracies, marker='s', linestyle=':', label='Sanskrit BERT (from Scratch)')
    
    plt.title('Model Accuracy Comparison on Root Commentary Classification')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accuracy comparison between two models.")
    parser.add_argument("--our_model_dir", type=str, required=True, help="Directory containing the log_history.json for our model.")
    parser.add_argument("--baseline_model_dir", type=str, required=True, help="Directory containing the log_history.json for the baseline model.")
    parser.add_argument("--from_scratch_model_dir", type=str, required=True, help="Directory containing the log_history.json for the from-scratch model.")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the output plot image.")
    
    args = parser.parse_args()

    our_log_path = Path(args.our_model_dir) / "log_history.json"
    baseline_log_path = Path(args.baseline_model_dir) / "log_history.json"
    from_scratch_log_path = Path(args.from_scratch_model_dir) / "log_history.json"
    output_path = Path(args.output_image)
    
    plot_comparison(our_log_path, baseline_log_path, from_scratch_log_path, output_path)
