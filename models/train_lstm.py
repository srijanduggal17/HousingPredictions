model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(num_epochs):
    # zero out gradient between epochs
    optimizer.zero_grad()
    
    # init hidden state
    model.hidden = model.init_hidden()
    
    # forward pass
    y_pred = model(X_train)
    
    # calculate loss
    loss = loss_function(y_pred, y_train)
    if i % 100 == 0:
        print("Epoch: ", i, "MSE: ", loss.item())

    # backward pass
    loss.backward()
    
    # update params
    optimizer.step()