

def test():
  # distance matrix
  # D_{ij} = ||x_i||_^2 + ||x_j||_2^2 - 2 x_i^Tx_j
  sum_row = np.sum(X ** 2, axis = 1)
  xxt = np.dot(X, X.transpose())
  dist_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * xxt
