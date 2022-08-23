def median_filter(pred, window_size):
    median_pred = []
    for start in range(0, len(pred) - window_size + 1):
        tmp = []
        for i in range(start, start + window_size):
            tmp.append(pred[i])
        tmp = sorted(tmp)
        median_label = tmp[window_size // 2]
        if start == 0 or start == len(pred) - window_size:
            for i in range(window_size // 2 + 1):
                median_pred.append(median_label)
        else:
            median_pred.append(median_label)
    return median_pred
