import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df):
    """Plot target variable distribution"""
    plt.figure(figsize=(6,6))
    ax = sns.countplot(df['default.payment.next.month'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom')
    plt.xticks([0,1], labels=["Not Defaulted", "Defaulted"])
    plt.title("Target Distribution")
    plt.show()

def plot_demographic_analysis(df):
    """Plot demographic distributions"""
    # Gender distribution
    plt.figure(figsize=(6,6))
    ax = sns.countplot('SEX', hue='default.payment.next.month', data=df)
    # ... rest of your plotting code