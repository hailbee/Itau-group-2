import argparse
import ast
import torch
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer
from model_utils.models.learning.siamese import SiameseModelPairs

def main():
    parser = argparse.ArgumentParser(description='CLIP-based text similarity training and evaluation')
    parser.add_argument('--mode', type=str, 
                      choices=['train', 'baseline', 'evaluate_saved', 'ensemble'], 
                      required=True,
                      help='Mode to run')
    parser.add_argument('--optuna', type=str, choices=[True, False], default=True,
                      help='Optuna hyperparameter optimization')
    parser.add_argument('--training_filepath', type=str,
                      help='Path to training data (for training modes)')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data (CSV or Parquet with fraudulent_name, real_name, label)')
    parser.add_argument('--baseline_model', type=str, choices=['clip', 'coca', 'flava', 'siglip', 'openclip', 'all'], default='clip',
                      help='Baseline model to test (for baseline mode)')
    parser.add_argument('--backbone', type=str, choices=['clip', 'coca', 'flava', 'siglip'], default='clip',
                      help='Vision-language backbone to use (clip, siglip, flava, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--medium_filepath', type=str,
                      help='Path to medium data (optional)')
    parser.add_argument('--easy_filepath', type=str,
                      help='easy to medium data (optional)')
    parser.add_argument('--external', action='store_true', default=False,
                      help='If set, evaluate on an external pairwise dataset (no reference set, only test_filepath required)')
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
    
    # Ensemble mode parameters
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the saved .pt model file (default: log_dir/best_model_siglip_pair.pt)')
    parser.add_argument('--ensemble_output_dir', type=str, default='ensemble_results',
                      help='Directory to save ensemble results (default: ensemble_results)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_siamese_model(mode, backbone_name, embedding_dim=512, projection_dim=128, device=None):
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type=backbone_name, batch_size=1, device=device)
        backbone_module = tester.model_wrapper  # Use the wrapper, not .model
        assert hasattr(backbone_module, 'encode_text'), f"Backbone {type(backbone_module)} does not have encode_text"
        return SiameseModelPairs(embedding_dim, projection_dim, backbone=backbone_module)

    
    if args.mode == 'evaluate_saved':
        print("Loading saved model for evaluation...")
        # Load backbone
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type=args.backbone, batch_size=1, device=device)
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
        # Single training run
        model = get_siamese_model('pair', args.backbone, embedding_dim=512, projection_dim=128, device=device).to(device)
        
        # Get appropriate loss class
        from model_utils.loss.pair_losses import CosineLoss
        criterion = CosineLoss(margin=0.5)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dataloaders
        import pandas as pd
        from torch.utils.data import DataLoader
        
        # Load training data
        dataframe = pd.read_parquet(args.training_filepath)
        
        # Create appropriate dataset and dataloader based on model type

        from utils.data import TextPairDataset
        dataset = TextPairDataset(dataframe)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # Create warmup dataloader if warmup filepath is provided
        easy_loader = None
        medium_loader = None
        if args.easy_filepath and args.medium_filepath:
            medium_dataframe = pd.read_parquet(args.medium_filepath)
            easy_dataframe = pd.read_parquet(args.easy_filepath)
            
            from utils.data import TextPairDataset
            medium_dataset = TextPairDataset(medium_dataframe)
            easy_dataset = TextPairDataset(easy_dataframe)
            medium_loader = DataLoader(medium_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            easy_loader = DataLoader(easy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)


        ### here: pass in the model_type
        trainer = Trainer(model, criterion, optimizer, device, model_type='pair')
        trainer.train(
            dataloader=dataloader,
            test_filepath=args.test_filepath,
            mode='pair',
            epochs=args.epochs,
            medium_loader=medium_loader,
            easy_loader=easy_loader,
            curriculum=args.curriculum,
            validate_filepath=args.validate_filepath,
            plot=args.plot
        )

    elif args.optuna == True:
        # Advanced hyperparameter optimization
        optimizer = UnifiedHyperparameterOptimizer(args.backbone, device=device, log_dir=args.log_dir)
        
        # Prepare optimization parameters
        opt_params = {
            'n_trials': args.n_trials,
            'sampler': args.sampler,
            'pruner': args.pruner if args.pruner != 'none' else None,
            'study_name': args.study_name,
            'epochs': args.epochs,
        }
        
        results = optimizer.optimize(
            method='optuna',
            training_filepath=args.training_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            curriculum=args.curriculum,
            **opt_params,
            validate_filepath=args.validate_filepath
        )


    elif args.mode == 'ensemble':
        # Ensemble mode - combine pre-trained model with fuzzy string matching features
        print("Running ensemble mode...")
        
        # Validate required arguments for ensemble mode
        if not args.training_filepath:
            print("Error: --training_filepath is required for ensemble mode")
            return
        
        if not args.test_filepath:
            print("Error: --test_filepath is required for ensemble mode")
            return
        
        # Use same default model path as evaluate_saved mode
        if not args.model_path:
            args.model_path = args.log_dir + "/best_model_siglip_pair.pt"
            print(f"Using default model path: {args.model_path}")
        
        # Import ensemble pipeline
        from scripts.ensemble_pipeline import EnsemblePipeline
        
        # Run ensemble pipeline
        try:
            pipeline = EnsemblePipeline(
                model_path=args.model_path,
                backbone=args.backbone,
                batch_size=args.batch_size
            )
            
            # Run the pipeline
            results = pipeline.run_pipeline(
                training_filepath=args.training_filepath,
                test_filepath=args.test_filepath,
                output_dir=args.ensemble_output_dir
            )
            
            print(f"\nEnsemble pipeline completed successfully!")
            print(f"Results saved to: {args.ensemble_output_dir}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"AUC: {results['auc']:.4f}")
            
        except Exception as e:
            print(f"Error running ensemble pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return

if __name__ == '__main__':
    main() 
