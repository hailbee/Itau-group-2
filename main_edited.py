import argparse
import ast
import torch
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer

# python3 main_edited.py --mode train --optuna False --training_filepath "/Users/a../Downloads/train_pairs_with_siglip_embeddings.parquet" --test_filepath "/Users/a../Downloads/test_pairs_with_siglip_embeddings.parquet" 
# same test/train file but just to see if training works

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a Siamese model for business name matching.')
    parser.add_argument('--mode', type=str, 
                      choices=['train', 'evaluate_saved'], 
                      required=True,
                      help='Mode to run: train or evaluate_saved')
    parser.add_argument('--optuna', type=str, choices=['True', 'False'], default='True',
                      help='Optuna hyperparameter optimization (True/False)')
    parser.add_argument('--training_filepath', type=str,
                      help='Path to training data (for training modes)')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data (CSV or Parquet with fraudulent_name, real_name, label)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--validate_filepath', type=str, default=None, help='Path to validation data file (CSV or Parquet). Used for mid-training and end-of-training validation.')
    parser.add_argument('--plot', action='store_true', help='If set, plot ROC and confusion matrices during evaluation')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--curriculum', type=str, default=None,
                      help='Curriculum learning mode')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/Project_2_Business_Names/Summer 2025/code',
                      help='Directory to save results')
    
    # Hyperparameter optimization parameters
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of trials for optimization methods')
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'], default='tpe',
                      help='Sampler for Optuna optimization')
    parser.add_argument('--pruner', type=str, choices=['median', 'hyperband', 'none'], default='median',
                      help='Pruner for Optuna optimization')
    parser.add_argument('--study_name', type=str,
                      help='Study name for Optuna optimization')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'evaluate_saved':
        print("Loading saved model for evaluation...")
        # Load backbone
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type='siglip', batch_size=1, device=device)
        backbone_module = tester.model_wrapper  # must have .encode_text
        
        # Load your model with matching dimensions
        model = SiameseModelPairs(embedding_dim=768, projection_dim=768, backbone=backbone_module).to(device)

        # Load saved weights
        state_dict = torch.load(args.log_dir + "/best_model_siglip_pair.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Evaluate
        evaluator = Evaluator(model, batch_size=args.batch_size, model_type='pair')
        results_df, metrics = evaluator.evaluate(args.test_filepath, plot=args.plot)

        print("\n Evaluation complete. Results:")
        for k, v in metrics.items():
            if k != 'roc_curve':
                print(f"{k}: {v}")
        
        # Save results_df to computer
        import pandas as pd
        from datetime import datetime
        import os
        
        # Create output directory if it doesn't exist
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluation_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save results_df to CSV
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")
        
        # Also save metrics to a separate file
        metrics_filename = f"evaluation_metrics_{timestamp}.json"
        metrics_filepath = os.path.join(output_dir, metrics_filename)
        
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_np(obj):
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(v) for v in obj]
            elif hasattr(obj, 'item') and callable(obj.item):
                return obj.item()
            else:
                return obj
        
        metrics_serializable = convert_np(metrics)
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"Metrics saved to: {metrics_filepath}")

    elif args.mode == 'train':
        if args.optuna == 'True':
            # OPTUNA TRAINING
            optimizer = UnifiedHyperparameterOptimizer(
                'pairwise_contrastive',
                device=device,
                log_dir=args.log_dir
            )

            results = optimizer.optimize(
                method='optuna',
                training_filepath=args.training_filepath,
                test_filepath=args.test_filepath,
                mode='pair',
                loss_type='contrastive',
                epochs=args.epochs,
                n_trials=args.n_trials,
                sampler=args.sampler,
                pruner=args.pruner if args.pruner != 'none' else None,
                validate_filepath=args.validate_filepath,
                curriculum=args.curriculum
            )

        else:
            # Single training run
            from model_utils.models.learning.siamese import SiameseEmbeddingModel

            model = SiameseEmbeddingModel(
                embedding_dim=768,
                hidden_dim=256,
                out_dim=128
            ).to(device)

            # Get appropriate loss class
            from model_utils.loss.pair_losses import ContrastiveLoss
            criterion = ContrastiveLoss(margin=1.0)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Create dataloaders
            import pandas as pd
            from torch.utils.data import DataLoader
            
            # Load training data
            dataframe = pd.read_parquet(args.training_filepath)
            
            # Create appropriate dataset and dataloader based on model type

            from utils.data import EmbeddingPairDataset
            dataset = EmbeddingPairDataset(dataframe)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

            ### here: pass in the model_type
            trainer = Trainer(model, criterion, optimizer, device, model_type='pair')
            trainer.train(
                dataloader=dataloader,
                mode='pair',
                epochs=args.epochs,
                validate_filepath=args.validate_filepath,
            )

if __name__ == '__main__':
    main() 
