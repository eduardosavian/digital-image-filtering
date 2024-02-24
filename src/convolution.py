import numpy as np
import cv2 as cv


def valid_mask_shape(mask: np.ndarray) -> bool:
  is_odd = lambda x: (x % 2) != 0
  return (mask.shape[0] == mask.shape[1]) and is_odd(mask.shape[0])


def mat_bounds_check(row, col, height, width) -> bool:
  ok = (col < width) and (col > 0) and (row < height) and (row > 0)
  return ok


def convolution(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
  assert valid_mask_shape(mask)

  N = mask.shape[0] // 2
  border_img = cv.copyMakeBorder(img, N,N,N,N, cv.BORDER_CONSTANT, value=(0,0,0))
  res = np.zeros( img.shape )
  HEIGHT, WIDTH = border_img.shape

  for row in range(N, HEIGHT - N):
    for col in range(N, WIDTH - N):
      slice = border_img[(row-N):(row+N+1), (col-N):(col+N+1)]
      total = np.clip(0, np.sum(mask * slice), 1.0)
      res[row - N, col - N] = total


  return res

