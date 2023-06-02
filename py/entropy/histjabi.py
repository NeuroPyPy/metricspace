import numpy as np


def histjabi(cvec):
    """
        Calculates the jackknife bias that should be added to the naive plugin estimate, in bits.

        Args:
            cvec (numpy.ndarray): Vector of counts.

        Returns:
            float: Jackknife bias in bits.
        """
    cvec = np.reshape(cvec, (np.prod(cvec.shape), 1))
    if np.max(cvec) <= 1:
        h = 0
        return h
    nsamps = np.sum(cvec)
    cv2 = cvec[cvec >= 2]
    jdev = np.log((nsamps - 1) / nsamps) + (1 / (nsamps * (nsamps - 1))) * np.sum(
        cv2 * (cv2 - 1) * np.log(cv2 / (cv2 - 1)))
    h = -(nsamps - 1) * jdev / np.log(2)  # first-order correction
    return h
