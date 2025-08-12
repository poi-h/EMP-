import matplotlib.pyplot as plt

def signal(x_list, y_list, labels=None, title='多曲线图', xlabel='X', ylabel='Y', save_path=None):
    """
    多条曲线画图工具
    参数:
        x_list: x轴数据列表（每个元素为一条曲线的x数据）
        y_list: y轴数据列表（每个元素为一条曲线的y数据）
        labels: 每条曲线的标签（可选）
        title: 图标题
        xlabel: x轴标签
        ylabel: y轴标签
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(8, 5))
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        label = labels[i] if labels and i < len(labels) else f'curve_{i+1}'
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()