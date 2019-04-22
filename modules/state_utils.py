def normalize_state(self, x):
    x = (x - self.state_min_val) / (self.state_max_val - self.state_min_val)
    return x

def average(x):
    return sum(x) / len(x)