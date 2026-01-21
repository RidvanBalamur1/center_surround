import torch
import torch.nn as nn


def train_one_epoch(model, dataloader, optimizer, criterion, device, use_regularizer=True):
    model.train()
    total_loss = 0
    
    for images, responses in dataloader:
        images, responses = images.to(device), responses.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, responses)
        
        # Add regularizer if model has one
        if use_regularizer and hasattr(model, 'regularizer'):
            loss = loss + model.regularizer()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, responses in dataloader:
            images, responses = images.to(device), responses.to(device)
            output = model(images)
            loss = criterion(output, responses)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(model, dataloaders, num_epochs, lr, device, use_regularizer=True, patience=15):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.PoissonNLLLoss(log_input=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device, use_regularizer)
        val_loss = evaluate(model, dataloaders['validation'], criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
