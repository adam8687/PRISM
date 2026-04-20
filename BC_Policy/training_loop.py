import torch.optim as optim

optimizer = optim.Adam(policy.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    total_loss = 0
    for obs_batch, act_batch in loader:
        obs_batch = obs_batch.cuda()
        act_batch = act_batch.cuda()

        pred_actions = policy(obs_batch)
        loss = nn.functional.mse_loss(pred_actions, act_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch:3d} | Loss: {total_loss/len(loader):.4f}")