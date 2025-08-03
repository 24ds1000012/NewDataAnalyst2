import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_plot(df, x, y, regression=True):
    plt.figure()
    plt.scatter(df[x], df[y])
    if regression:
        import numpy as np
        m, b = np.polyfit(df[x], df[y], 1)
        plt.plot(df[x], m*df[x] + b, color="red", linestyle="dotted")
    plt.xlabel(x)
    plt.ylabel(y)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
    