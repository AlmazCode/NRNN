"""Modern CLI for training and generating text with TextRNN.

Examples:
    Train a model:
        python rnn.py train data.txt -o my_model --epochs 30
        python rnn.py train data.txt --quick  # Fast training preset
    
    Generate text:
        python rnn.py generate my_model "Once upon a time"
        python rnn.py generate my_model "Start" --auto  # Auto-stop generation
        python rnn.py generate my_model --interactive  # Interactive mode
    
    Model info:
        python rnn.py info my_model
"""

import argparse
import sys
import os   
from pathlib import Path

from rnn import TextRNN
from tools import console


def add_architecture_args(parser, include_dropout=True):
    """Add common architecture arguments to a parser."""
    arch = parser.add_argument_group('Architecture')
    arch.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    arch.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    arch.add_argument('--n-layers', type=int, default=2, help='Number of LSTM layers')
    if include_dropout:
        arch.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')


def _generate_and_print(rnn, prompt, args):
    """Generate text using `rnn` and print/save the result."""
    length = -1 if args.auto else args.length
    text = rnn.generate_text(
        prompt,
        length=length,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"{console.Color.Fore.GREEN}{text}{console.Color.Style.RESET}\n")

    if getattr(args, 'output_file', None):
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        console.Console.log(f"Saved to {args.output_file}")


def train_command(args):
    """Train a new model."""
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        console.Console.error(f"File not found: {input_file}")
        sys.exit(1)
    
    console.Console.log("TRAINING NEW MODEL")
    
    # Load text
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        console.Console.log(f"Loaded text from {input_file}")
        console.Console.log(f"Text length: {len(text):,} characters")
    except Exception as e:
        console.Console.error(f"Failed to read file: {e}")
        sys.exit(1)
    
    # Quick preset
    if args.quick:
        args.epochs = 10
        args.batch_size = 256
        args.hidden_dim = 128
        console.Console.log("Using quick training preset (10 epochs, larger batches)")

    # Determine output path early so TextRNN has an identifier
    output_path = args.output or input_file.stem + "_model"

    # Initialize model
    console.Console.log(f"Architecture: {args.n_layers} layers, {args.hidden_dim} hidden units")
    rnn = TextRNN(
        model_path=output_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Prepare data
    dataset = rnn.prepare_data(
        text, 
        seq_length=args.seq_length,
        add_end_tokens=not args.no_end_tokens
    )
    
    # Train
    rnn.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        early_stop_patience=args.patience,
        resume=args.resume,
        resume_from=args.resume_from
    )
    
    # Save
    rnn.save_model(output_path)


def generate_command(args):
    """Generate text from a trained model."""
    model_path = args.model_path
    
    # Check if model exists
    if not Path(f"assets/trained/{model_path}").exists():
        console.Console.log(f"Model not found: {model_path}")
        console.Console.log("Train a model first with: python rnn.py train <data.txt>")
        sys.exit(1)
    
    # Load model
    console.Console.log(f"Loading model from {model_path}...")
    rnn = TextRNN(
        model_path=model_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )
    
    try:
        rnn.load_model(model_path)
        console.Console.log("Model loaded successfully")
    except Exception as e:
        console.Console.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        console.Console.log("INTERACTIVE GENERATION MODE")
        console.Console.log("Type your prompt and press Enter. Type 'quit' to exit.\n")
        
        while True:
            try:
                prompt = input(f"{console.Color.Fore.YELLOW}Prompt: {console.Color.Style.RESET}").strip()
                if prompt.lower() in ('quit', 'exit', 'q'):
                    break
                if not prompt:
                    continue
                
                print(f"{console.Color.Fore.CYAN}Generating...{console.Color.Style.RESET}")
                try:
                    _generate_and_print(rnn, prompt, args)
                except Exception as e:
                    console.Console.error(f"Generation failed: {e}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                console.Console.error(f"Generation failed: {e}")
        
        return
    
    # Single generation
    if not args.start_text:
        console.Console.error("Provide starting text with positional argument or use --interactive")
        sys.exit(1)
    
    console.Console.log("GENERATING TEXT")
    console.Console.log(f"Start: '{args.start_text}'")
    console.Console.log(f"Temperature: {args.temperature}")
    
    if args.auto:
        console.Console.log("Mode: Auto-stop (will stop at END token)")
    else:
        console.Console.log(f"Length: {args.length} characters")
    
    print()
    
    try:
        _generate_and_print(rnn, args.start_text, args)
    except Exception as e:
        console.Console.error(f"Generation failed: {e}")
        sys.exit(1)


def info_command(args):
    """Display model information."""
    model_path = args.model_path
    model_file = Path(f"trained/{model_path}/model.pth")
    if not model_file.exists():
        console.Console.error(f"Model not found: {model_file}")
        sys.exit(1)
    
    console.Console.log("MODEL INFORMATION")
    
    rnn = TextRNN(model_path=model_path)
    try:
        rnn.load_model(model_path)
    except Exception as e:
        console.Console.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    print(f"{console.Color.Fore.BOLD}Model Path:{console.Color.Style.RESET} {model_path}")
    print(f"{console.Color.Fore.BOLD}Vocabulary Size:{console.Color.Style.RESET} {rnn.vocab_size}")
    print(f"{console.Color.Fore.BOLD}Hidden Dimension:{console.Color.Style.RESET} {rnn.hidden_dim}")
    print(f"{console.Color.Fore.BOLD}Embedding Dimension:{console.Color.Style.RESET} {rnn.embedding_dim}")
    print(f"{console.Color.Fore.BOLD}Number of Layers:{console.Color.Style.RESET} {rnn.n_layers}")
    print(f"{console.Color.Fore.BOLD}END Token:{console.Color.Style.RESET} '{rnn.END_TOKEN}' (index: {rnn.end_token_idx})")
    print(f"{console.Color.Fore.BOLD}Device:{console.Color.Style.RESET} {rnn.device}")
    
    # Model size
    model_size = os.path.getsize(model_file) / (1024 * 1024)
    print(f"{console.Color.Fore.BOLD}Model Size:{console.Color.Style.RESET} {model_size:.2f} MB")
    
    # Sample characters
    sample_chars = list(rnn.char_to_idx.keys())[:20]
    print(f"{console.Color.Fore.BOLD}Sample Characters:{console.Color.Style.RESET} {repr(''.join(sample_chars))}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Modern CLI for TextRNN training and generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ==================== TRAIN COMMAND ====================
    train_parser = subparsers.add_parser(
        'train',
        help='Train a new model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        'input_file',
        help='Input text file for training'
    )
    train_parser.add_argument(
        '-o', '--output',
        help='Output model name (without extension, default: input_file_model)'
    )
    train_parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick training preset (fewer epochs, larger batches)'
    )
    
    # Architecture
    add_architecture_args(train_parser)
    
    # Training
    train_group = train_parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_group.add_argument('--batch-size', type=int, default=128, help='Batch size')
    train_group.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_group.add_argument('--seq-length', type=int, default=100, help='Sequence length')
    train_group.add_argument('--grad-clip', type=float, default=5.0, help='Gradient clipping threshold')
    train_group.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
    train_group.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    train_group.add_argument('--no-end-tokens', action='store_true', help='Disable END token insertion')
    train_group.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint in model dir')
    train_group.add_argument('--resume-from', help='Resume from specific checkpoint path (relative to trained/ or absolute)')
    
    train_parser.set_defaults(func=train_command)
    
    # ==================== GENERATE COMMAND ====================
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate text from a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    gen_parser.add_argument(
        'model_path',
        help='Path to saved model (without extension)'
    )
    gen_parser.add_argument(
        'start_text',
        nargs='?',
        default='',
        help='Starting text for generation (optional in interactive mode)'
    )
    
    # Generation modes
    mode_group = gen_parser.add_argument_group('Generation Mode')
    mode_group.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive generation mode'
    )
    mode_group.add_argument(
        '-a', '--auto',
        action='store_true',
        help='Auto-stop generation at END token'
    )
    
    # Generation parameters
    gen_group = gen_parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('-l', '--length', type=int, default=300, help='Length of generated text')
    gen_group.add_argument('--max-length', type=int, default=1000, help='Max length for auto-stop mode')
    gen_group.add_argument('-t', '--temperature', type=float, default=0.8, help='Sampling temperature')
    gen_group.add_argument('--top-k', type=int, help='Top-k sampling (default: None)')
    gen_group.add_argument('--output-file', help='Save generated text to file')
    
    # Architecture (must match training)
    add_architecture_args(gen_parser, include_dropout=False)
    
    gen_parser.set_defaults(func=generate_command)
    
    # ==================== INFO COMMAND ====================
    info_parser = subparsers.add_parser(
        'info',
        help='Display model information'
    )
    info_parser.add_argument(
        'model_path',
        help='Path to saved model (without extension)'
    )
    info_parser.set_defaults(func=info_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(f"\n{console.Color.Fore.YELLOW}Interrupted by user{console.Color.Style.RESET}")
        sys.exit(130)
    except Exception as e:
        console.Console.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    os.system("cls")
    main()