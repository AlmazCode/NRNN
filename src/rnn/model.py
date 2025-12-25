from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import pickle
import re
import os

from tools.console import Console


class CharDataset(Dataset):
    """Dataset for character-level text generation.
    
    Args:
        text: List of integer-encoded characters
        seq_length: Length of input sequences
    """
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        # keep underlying data as a single torch tensor to avoid
        # allocating new tensors for every __getitem__ call
        self.text = torch.tensor(text, dtype=torch.long)
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        # Return views/slices of the prebuilt tensor (no new allocations)
        return (
            self.text[idx:idx + self.seq_length],
            self.text[idx + self.seq_length]
        )


class CharRNN(nn.Module):
    """Character-level RNN with LSTM architecture.
    
    Args:
        vocab_size: Size of the character vocabulary
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden states
        n_layers: Number of LSTM layers
        dropout: Dropout probability (applied when n_layers > 1)
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, n_layers=2, dropout=0.3):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Optional hidden state tuple (h, c)
            
        Returns:
            output: Logits of shape (batch_size, vocab_size)
            hidden: Updated hidden state tuple
        """
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state with zeros.
        
        Args:
            batch_size: Size of the batch
            device: Device to create tensors on
            
        Returns:
            Tuple of (h0, c0) hidden states
        """
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        )


class TextRNN:
    """Wrapper class for training and generating text with CharRNN.
    
    Args:
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden states
        n_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(self, model_path, embedding_dim=256, hidden_dim=128, n_layers=2, dropout=0.3):

        self.model_name = os.path.split(model_path)[-1]

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.END_TOKEN = '<END>'
        self.end_token_idx = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Console.log(f"Using device: {self.device}")
        
    def prepare_data(self, text, seq_length=100, add_end_tokens=True, end_pattern=r'\n\n'):
        """Prepare text data for training.
        
        Args:
            text: Raw text string
            seq_length: Length of training sequences
            add_end_tokens: Whether to add END tokens to text
            end_pattern: Regex pattern where to insert END tokens (default: double newlines)
            
        Returns:
            CharDataset object ready for training
        """
        # Add END tokens at logical boundaries (paragraphs by default)
        if add_end_tokens:
            text = re.sub(end_pattern, f'{self.END_TOKEN}\\g<0>', text)
        
        # Build vocabulary including END token
        chars = sorted(list(set(text)))
        if self.END_TOKEN not in chars:
            chars.append(self.END_TOKEN)
        
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.end_token_idx = self.char_to_idx[self.END_TOKEN]
        
        Console.log(f"Unique characters: {self.vocab_size}")
        Console.log(f"Text length: {len(text)}")
        Console.log(f"END token index: {self.end_token_idx}")
        
        # Convert text to integer indices
        text_as_int = [self.char_to_idx[c] for c in text]
        
        dataset = CharDataset(text_as_int, seq_length)
        Console.log(f"Number of sequences: {len(dataset)}")
        
        return dataset
    
    def build_model(self):
        """Build the CharRNN model.
        
        Returns:
            CharRNN model instance
        """
        self.model = CharRNN(
            self.vocab_size, 
            self.embedding_dim, 
            self.hidden_dim,
            self.n_layers,
            self.dropout
        ).to(self.device)
        
        return self.model
    
    def train(self, dataset, epochs=50, batch_size=128, lr=0.001, 
              grad_clip=5.0, val_split=0.1, early_stop_patience=5,
              num_workers=0, use_amp=None, resume=False, resume_from=None):
        """Train the model with validation and early stopping.
        
        Args:
            dataset: CharDataset for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            grad_clip: Gradient clipping threshold
            val_split: Fraction of data to use for validation
            early_stop_patience: Number of epochs to wait before early stopping
        """
        if self.model is None:
            self.build_model()
        
        # Split dataset into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # DataLoader performance options
        if use_amp is None:
            use_amp = (self.device.type == 'cuda')

        dataloader_kwargs = dict(
            batch_size=batch_size,
            pin_memory=(self.device.type == 'cuda'),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )

        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

        # Gradient scaler for automatic mixed precision (AMP)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_loss = float('inf')
        patience_counter = 0

        # Optionally resume from a saved checkpoint
        start_epoch = 0
        if resume or resume_from:
            try:
                ckpt = self.load_checkpoint(resume_from)
                # load model state
                state = ckpt.get('model_state')
                if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
                    new_state = {k.replace('module.', '', 1): v for k, v in state.items()}
                    state = new_state
                self.model.load_state_dict(state)

                # load optimizer, scaler, scheduler if present
                if 'optimizer_state' in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state'])
                    except Exception:
                        Console.warning("Failed to fully restore optimizer state")
                if 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
                    try:
                        scaler.load_state_dict(ckpt['scaler_state'])
                    except Exception:
                        Console.warning("Failed to restore AMP scaler state")
                if 'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state'])
                    except Exception:
                        Console.warning("Failed to restore scheduler state")

                start_epoch = ckpt.get('epoch', 0) + 1
                best_val_loss = ckpt.get('best_val_loss', best_val_loss)
                Console.log(f"Resuming training from epoch {start_epoch}")
            except FileNotFoundError:
                Console.warning("No checkpoint found to resume; starting fresh")
        
        Console.log("\nStarting training...")
        Console.log(f"Train size: {train_size}, Val size: {val_size}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                # Mixed precision if enabled
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output, _ = self.model(x)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                else:
                    output, _ = self.model(x)
                    loss = criterion(output, y)
                    loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                batch_size_actual = y.size(0)
                train_loss_sum += loss.item() * batch_size_actual

                # Calculate accuracy from logits
                predicted = torch.argmax(output, dim=1)
                train_total += batch_size_actual
                train_correct += (predicted == y).sum().item()
                
                if batch_idx % 50 == 0:
                    Console.log(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')
            
            # Validation phase
            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            output, _ = self.model(x)
                            loss = criterion(output, y)
                    else:
                        output, _ = self.model(x)
                        loss = criterion(output, y)

                    batch_size_actual = y.size(0)
                    val_loss_sum += loss.item() * batch_size_actual
                    predicted = torch.argmax(output, dim=1)
                    val_total += batch_size_actual
                    val_correct += (predicted == y).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss_sum / train_total if train_total else float('nan')
            avg_val_loss = val_loss_sum / val_total if val_total else float('nan')
            train_acc = 100 * train_correct / train_total if train_total else 0.0
            val_acc = 100 * val_correct / val_total if val_total else 0.0
            
            Console.log(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping / checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model weights+vocab
                self.save_model(f"{self.model_name}/backup")
                # Save a best checkpoint with optimizer/scaler/scheduler state
                try:
                    self.save_checkpoint(optimizer, epoch+start_epoch, scaler=scaler, best_val_loss=best_val_loss,
                                         filepath=f"{self.model_name}/backup/checkpoint_best.pt",
                                         scheduler=scheduler)
                except Exception as e:
                    Console.warning(f"Failed to save best checkpoint: {e}")

                Console.log(f"New best model saved! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    Console.warning(f"Early stopping triggered after {epoch+1} epochs")
                    # Load last backup model
                    try:
                        self.load_model(f"{self.model_name}/backup")
                        # try restore checkpoint info for information
                        ckpt = self.load_checkpoint(f"{self.model_name}/backup/checkpoint_best.pt")
                        Console.log(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")
                    except Exception:
                        # best-effort - ignore failures here
                        pass
                    break
            # Save latest checkpoint each epoch so training can resume from recent state
            try:
                self.save_checkpoint(optimizer, epoch+start_epoch, scaler=scaler, best_val_loss=best_val_loss,
                                     filepath=f"{self.model_name}/checkpoint_latest.pt", scheduler=scheduler)
            except Exception as e:
                Console.warning(f"Failed to save latest checkpoint: {e}")
    
    def generate_text(self, start_string, length=-1, max_length=1000, 
                     temperature=1.0, top_k=None):
        """Generate text from a starting string.
        
        Args:
            start_string: Initial text to start generation
            length: Number of characters to generate. If -1, auto-stop at END token
            max_length: Maximum length when using auto-stop
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely characters
            
        Returns:
            Generated text string
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        self.model.eval()
        auto_stop = (length == -1)
        if auto_stop:
            length = max_length
        
        # Convert start string to indices and prime the hidden state.
        chars = [self.char_to_idx.get(c, 0) for c in start_string]

        generated = list(start_string)
        hidden = None

        # Initial input token: last character of prompt (or 0 if empty)
        if chars:
            input_token = torch.tensor([[chars[-1]]], dtype=torch.long).to(self.device)
            with torch.no_grad():
                # Prime the model with the full prompt to get correct hidden state
                full_input = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(self.device)
                _, hidden = self.model(full_input, hidden)
        else:
            input_token = torch.tensor([[0]], dtype=torch.long).to(self.device)

        # Validate temperature
        if temperature <= 0:
            raise ValueError('`temperature` must be > 0')

        with torch.no_grad():
            for i in range(length):
                output, hidden = self.model(input_token, hidden)

                # output shape: (1, vocab_size) => logits
                logits = output / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=1)
                    mask = torch.full_like(logits, -float('Inf'))
                    mask.scatter_(1, indices, values)
                    logits = mask

                probs = torch.softmax(logits, dim=1)

                # Sample next character
                next_char_idx = torch.multinomial(probs, 1).item()

                # Check for auto-stop
                if auto_stop and next_char_idx == self.end_token_idx:
                    Console.log(f"Auto-stopped at length {i} (END token reached)")
                    break

                # Skip adding END token to output
                if next_char_idx != self.end_token_idx:
                    generated.append(self.idx_to_char[next_char_idx])

                # Update input token for next step
                input_token = torch.tensor([[next_char_idx]], dtype=torch.long).to(self.device)
        
        return ''.join(generated)
    
    def save_model(self, filepath):
        """Save model weights and vocabulary.
        
        Args:
            filepath: Base path for saving files (without extension)
        """

        model_path = os.path.join("src", "assets", "trained", filepath, "model.pth")
        vocab_path = os.path.join("src", "assets", "trained", filepath, "vocab.pkl")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), model_path)
        with open(vocab_path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size,
                'end_token_idx': self.end_token_idx,
                'END_TOKEN': self.END_TOKEN,
                'hidden_dim': self.hidden_dim,
                "n_layers": self.n_layers,
                "embedding_dim": self.embedding_dim
            }, f)

        Console.log(f"Model saved to {filepath}")

    def save_checkpoint(self, optimizer, epoch, scaler=None, best_val_loss=None, filepath=None, scheduler=None):
        """Save a training checkpoint including optimizer and scaler state.

        ``filepath`` follows same convention as `save_model` (base path under `trained/`).
        If `filepath` ends with '.pt' it's treated as a full path relative to project root (or absolute).
        """
        # Default to a rolling latest checkpoint
        if filepath is None:
            checkpoint_path = os.path.join('src', 'assets', 'trained', self.model_name, 'checkpoint_latest.pt')
        elif filepath.endswith('.pt'):
            checkpoint_path = filepath if os.path.isabs(filepath) else os.path.join('src', 'assets', 'trained', filepath)
        else:
            checkpoint_path = os.path.join('src', 'assets', 'trained', filepath, 'checkpoint.pt')

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }

        torch.save(state, checkpoint_path)
        Console.log(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filepath=None):
        """Load a training checkpoint and return its dictionary.

        If ``filepath`` is None attempts to load trained/<model_name>/checkpoint.pt
        """
        # If no filepath provided, try latest -> fallback to legacy names -> backup best
        if filepath is None:
            candidates = [
                os.path.join('src', 'assets', 'trained', self.model_name, 'checkpoint_latest.pt'),
                os.path.join('src', 'assets', 'trained', self.model_name, 'checkpoint.pt'),
                os.path.join('src', 'assets', 'trained', self.model_name, 'backup', 'checkpoint_best.pt'),
            ]
            for p in candidates:
                if os.path.exists(p):
                    checkpoint_path = p
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found for model {self.model_name}")
        elif filepath.endswith('.pt'):
            checkpoint_path = filepath if os.path.isabs(filepath) else os.path.join('src', 'assets', 'trained', filepath)
        else:
            checkpoint_path = os.path.join('src', 'assets', 'trained', filepath, 'checkpoint.pt')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)
        return state
    
    def load_model(self, filepath):
        """Load model weights and vocabulary.
        
        Args:
            filepath: Base path for loading files (without extension)
        """

        model_path = os.path.join("assets", "trained", filepath, "model.pth")
        vocab_path = os.path.join("assets", "trained", filepath, "vocab.pkl")

        # Basic checks
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load vocabulary and model metadata
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.char_to_idx = vocab_data['char_to_idx']
            self.idx_to_char = vocab_data['idx_to_char']
            self.vocab_size = vocab_data['vocab_size']
            self.end_token_idx = vocab_data['end_token_idx']
            self.END_TOKEN = vocab_data['END_TOKEN']
            # update architecture params from saved file
            self.hidden_dim = vocab_data.get('hidden_dim', self.hidden_dim)
            self.n_layers = vocab_data.get('n_layers', self.n_layers)
            self.embedding_dim = vocab_data.get('embedding_dim', self.embedding_dim)

        # Build model to match saved architecture
        self.build_model()

        # Load state dict and handle possible DataParallel 'module.' prefixes
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            new_state = {}
            for k, v in state.items():
                new_state[k.replace('module.', '', 1)] = v
            state = new_state

        # Load parameters into model
        try:
            self.model.load_state_dict(state)
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Failed to load model state dict: {e}") from e

        self.model.to(self.device)
        self.model.eval()

        Console.log(f"Model loaded from {filepath}")