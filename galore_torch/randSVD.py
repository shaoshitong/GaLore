import torch

# ------------------------------------------------------------------------------
def rsvd(A, rank, n_oversamples=None, n_subspace_iters=None, return_range=False):
    """
    Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix (torch tensor).
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    B = torch.matmul(Q.T, A)
    U_tilde, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = torch.matmul(Q, U_tilde)

    # Truncate.
    U, S, Vt = U, S, Vt

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt

# ------------------------------------------------------------------------------

def find_range(A, n_samples, n_subspace_iters=None):
    """
    Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix (torch tensor).
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    m, n = A.shape
    O = torch.randn(n, n_samples, device=A.device, dtype=A.dtype)
    Y = torch.matmul(A, O)

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)

# ------------------------------------------------------------------------------

def subspace_iter(A, Y0, n_iters):
    """
    Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix (torch tensor).
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power iterations.
    """
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(torch.matmul(A.T, Q))
        Q = ortho_basis(torch.matmul(A, Z))
    return Q

# ------------------------------------------------------------------------------

def ortho_basis(M):
    """
    Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix (torch tensor).
    :return:  An orthonormal basis for M.
    """
    Q, _ = torch.linalg.qr(M)
    return Q


if __name__ == "__main__":
    # Create a random matrix A of size (m x n)
    m, n = 1000, 500
    A = torch.randn(m, n, device='cuda')  # 在cuda上创建随机矩阵A
    rank = 64
    U, S, Vt = rsvd(A, rank)
    A_approx = U @ torch.diag(S) @ Vt
    error = torch.norm(A - A_approx) / A.numel()
    print(f"Relative approximation error with A: {error.item():.6f}")
    
    U, S, Vt = torch.linalg.svd(A)
    A_approx2 = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
    error = torch.norm(A_approx - A_approx2) / A_approx.numel()
    print(f"Relative approximation error with vanila SVD: {error.item():.6f}")
    
    error = torch.norm(A - A_approx2) / A.numel()
    print(f"Relative approximation error with vanila SVD: {error.item():.6f}")
