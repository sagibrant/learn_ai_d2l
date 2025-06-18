import torch
import matplotlib.pyplot as plt

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5),
                  cmap='Reds'):
    """Display matrix heatmaps in a grid"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    print("num_rows, num_cols", num_rows, num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)  # <- use this instead of tight_layout
    print("fig, axes", fig, axes)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        print("i, (row_axes, row_matrices)", i, (row_axes, row_matrices))
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            print("j, (ax, matrix))", j, (ax, matrix))
            matrix_numpy = matrix.detach().numpy()
            pcm = ax.imshow(matrix_numpy, cmap=cmap)
            print("matrix_numpy, pcm", matrix_numpy, pcm)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    # Use all axes for colorbar placement
    fig.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.6)
    print("pcm, ax", pcm, axes.ravel().tolist())
    plt.show()


# Test: Identity attention matrix (1 batch, 1 head, 10x10 attention)
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
