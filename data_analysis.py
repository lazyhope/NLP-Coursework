import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RobertaTokenizer

# ------------------------------
# 1. File Paths
# ------------------------------
TRAIN_PATH = "data/whole_train_data.csv"

# ------------------------------
# 2. Load Data Function
# ------------------------------
def load_train_data():
    print("Loading training data...")
    # Load whole_train_data.csv which contains columns: par_id, country_code, keyword, text, label
    train_df = pd.read_csv(TRAIN_PATH)
    # Ensure label is integer
    train_df["label"] = train_df["label"].astype(int)
    print(f"Training examples: {len(train_df)}")
    return train_df

# ------------------------------
# 3. Analyze Label Distribution
# ------------------------------
# Define our colors for labels
COLOR_0 = "#94b2f7"  # Pastel blue for Label 0
COLOR_1 = "#f7a2a4"  # Pastel pink for Label 1

def analyze_label_distribution(df):
    label_counts = df["label"].value_counts().sort_index()
    print("Label distribution:")
    print(label_counts)
    
    # Use defined colors: assume labels are 0 and 1
    colors = [COLOR_0 if label == 0 else COLOR_1 for label in label_counts.index]
    
    plt.figure(figsize=(6,3))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette=colors)
    plt.xlabel("Label (0 = Non-PCL, 1 = PCL)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Label Distribution", fontsize=16)
    
    # Get the maximum count to adjust the top y-limit
    max_count = label_counts.values.max()
    plt.ylim(0, max_count + 1500)  # Add some margin above the tallest bar

    # Place text slightly above each bar
    for i, count in enumerate(label_counts.values):
        plt.text(
            i, 
            count + 200,      # Offset of 200 above the bar height
            f"{count}", 
            ha="center", 
            va="bottom", 
            fontsize=14
        )
    plt.tight_layout()
    plt.savefig("figures/label_distribution.png")
    plt.show()


# ------------------------------
# 4. Analyze Word Count
# ------------------------------
def analyze_word_count(df):
    # Create new column with text length (word count)
    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
    
    # Group by label and compute descriptive stats
    length_stats_by_label = df.groupby("label")["text_length"].describe().round(0).astype(int)
    print("\nText length statistics by label:")
    print(length_stats_by_label)
    
    # Boxplot of text length by label
    plt.figure(figsize=(6,3))
    sns.boxplot(x="label", y="text_length", data=df, palette="viridis")
    plt.title("How Labels Correlate With Text Length", fontsize=16)
    plt.xlabel("Label (0 = Non-PCL, 1 = PCL)", fontsize=14)
    plt.ylabel("Text Length (number of words)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/word_count_boxplot.png")
    plt.show()
    
    # Histogram of text length by label
    plt.figure(figsize=(12,5))
    sns.histplot(
        data=df,
        x="text_length",
        hue="label",
        bins=30,
        kde=True,
        palette="viridis",
        alpha=0.5
    )
    plt.title("Distribution of Text Length by Label", fontsize=16)
    plt.xlabel("Text Length (number of words)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/word_count_histogram.png")
    plt.show()

# ------------------------------
# 5. Analyze Token Count using RoBERTa Tokenizer
# ------------------------------
def analyze_token_count(df, tokenizer):
    # Get tokenized length for each text
    df["token_length"] = df["text"].apply(lambda x: len(tokenizer.tokenize(str(x))))
    
    # Group by label and compute descriptive stats
    token_stats_by_label = df.groupby("label")["token_length"].describe().round(0).astype(int)
    print("\nToken length statistics by label:")
    print(token_stats_by_label)
    
    # Boxplot of token length by label
    plt.figure(figsize=(6,3))
    sns.boxplot(x="label", y="token_length", data=df, palette="viridis")
    plt.title("Token Count Distribution by Label", fontsize=16)
    plt.xlabel("Label (0 = Non-PCL, 1 = PCL)", fontsize=14)
    plt.ylabel("Token Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/token_count_boxplot.png")
    plt.show()
    
    plt.figure(figsize=(8,5))
    sns.histplot(
        data=df,
        x="token_length",
        hue="label",
        bins=30,
        kde=True,
        palette=[COLOR_0, COLOR_1],
        alpha=0.5
    )
    plt.title("Token Count Distribution by Label", fontsize=16)
    plt.xlabel("Token Count", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/token_count_histogram.png")
    plt.show()


# ------------------------------
# 6. Analyze Keyword Counts and Percentages
# ------------------------------
def analyze_keyword_counts(df):
    # Count keyword occurrences for each label
    keyword_counts = df.groupby(["keyword", "label"]).size().reset_index(name="count")
    print("\nKeyword counts (head):")
    print(keyword_counts.head())

    # Pivot table: rows=keyword, columns=label
    keyword_pivot = keyword_counts.pivot(index="keyword", columns="label", values="count").fillna(0)
    
    # Dynamically rename columns based on unique labels
    unique_labels = sorted(keyword_pivot.columns)
    keyword_pivot.columns = [f"Label {label}" for label in unique_labels]

    # Calculate keyword percentages by label
    label_totals = keyword_pivot.sum(axis=0)  # total counts for each label
    keyword_percentage = keyword_pivot.divide(label_totals, axis=1) * 100

    # Plot side-by-side horizontal bars for percentages
    plot_keyword_percentage_sidebyside(keyword_percentage)

    # Plot stacked horizontal bars for raw counts
    plot_keyword_counts_stacked(keyword_pivot)

# ------------------------------
# 6A. Plot Keyword Percentage (Side-by-Side Horizontal Bars)
# ------------------------------
def plot_keyword_percentage_sidebyside(keyword_percentage):
    """
    Plots a horizontal bar chart of keyword percentages for each label,
    sorted descending by Label 1 (or the last label if Label 1 is not present),
    with numeric annotations.
    """
    # Determine target label for sorting: use "Label 1" if available, else last column.
    target_label = "Label 1" if "Label 1" in keyword_percentage.columns else keyword_percentage.columns[-1]
    keyword_percentage_sorted = keyword_percentage.sort_values(target_label, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(len(keyword_percentage_sorted))
    bar_height = 0.4

    labels = list(keyword_percentage_sorted.columns)
    if len(labels) >= 2:
        ax.barh(
            y_positions,
            keyword_percentage_sorted[labels[1]],
            height=bar_height,
            color=COLOR_1,
            label=labels[1]
        )
        ax.barh(
            y_positions + bar_height,
            keyword_percentage_sorted[labels[0]],
            height=bar_height,
            color=COLOR_0,
            label=labels[0]
        )
        for i, (kw, row) in enumerate(keyword_percentage_sorted.iterrows()):
            ax.text(row[labels[1]] + 0.5, i, f"{row[labels[1]]:.2f}%", va="center", fontsize=14)
            ax.text(row[labels[0]] + 0.5, i + bar_height, f"{row[labels[0]]:.2f}%", va="center", fontsize=14)
        ax.set_yticks(y_positions + bar_height / 2)
    else:
        ax.barh(
            y_positions,
            keyword_percentage_sorted[labels[0]],
            height=bar_height,
            color=COLOR_1,
            label=labels[0]
        )
        for i, (kw, row) in enumerate(keyword_percentage_sorted.iterrows()):
            ax.text(row[labels[0]] + 0.5, i, f"{row[labels[0]]:.2f}%", va="center", fontsize=14)
        ax.set_yticks(y_positions)
    
    ax.set_yticklabels(keyword_percentage_sorted.index, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion in Sentences (%)", fontsize=16)
    ax.set_title("Keyword Distribution by Label (Percentage)", fontsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/keyword_percentage_sidebyside.png")
    plt.show()

# ------------------------------
# 6B. Plot Keyword Counts (Stacked Horizontal Bars)
# ------------------------------
def plot_keyword_counts_stacked(keyword_counts):
    """
    Plots a stacked horizontal bar chart of keyword counts for each label,
    sorted in the same order as the percentages, with numeric annotations.
    """
    # For consistency, sort descending by Label 1 (or last column if not Label 1)
    target_label = "Label 1" if "Label 1" in keyword_counts.columns else keyword_counts.columns[-1]
    keyword_counts_sorted = keyword_counts.sort_values(target_label, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(len(keyword_counts_sorted))
    left = np.zeros(len(keyword_counts_sorted))
    for label in keyword_counts_sorted.columns:
        ax.barh(
            y_positions,
            keyword_counts_sorted[label],
            left=left,
            color=COLOR_0 if label == "Label 0" else COLOR_1,
            label=label
        )
        for i, value in enumerate(keyword_counts_sorted[label]):
            if value > 0:
                ax.text(left[i] + value/2, i, f"{int(value)}", ha="center", va="center", color="white", fontsize=14)
        left += keyword_counts_sorted[label].values

    ax.set_yticks(y_positions)
    ax.set_yticklabels(keyword_counts_sorted.index, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Count of Sentences", fontsize=16)
    ax.set_title("Keyword Counts by Label (Stacked)", fontsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/keyword_counts_stacked.png")
    plt.show()

# ------------------------------
# 6C. Analyze Keyword Counts
# ------------------------------
def analyze_keyword_counts(df):
    # Count keyword occurrences for each label
    keyword_counts = df.groupby(["keyword", "label"]).size().reset_index(name="count")
    print("\nKeyword counts (head):")
    print(keyword_counts.head())

    # Pivot table: rows=keyword, columns=label
    keyword_pivot = keyword_counts.pivot(index="keyword", columns="label", values="count").fillna(0)
    
    # Dynamically rename columns based on unique labels
    unique_labels = sorted(keyword_pivot.columns)
    keyword_pivot.columns = [f"Label {label}" for label in unique_labels]

    # Calculate keyword percentage by label
    label_totals = keyword_pivot.sum(axis=0)  # total counts for each label
    keyword_percentage = keyword_pivot.divide(label_totals, axis=1) * 100

    # **Plot vertical side-by-side bars** for keyword percentages
    plot_keyword_percentage_vertical(keyword_percentage)

    # Plot side-by-side horizontal bars for percentages
    plot_keyword_percentage_sidebyside(keyword_percentage)

    # Plot stacked horizontal bars for raw counts
    plot_keyword_counts_stacked(keyword_pivot)

def plot_keyword_percentage_vertical(keyword_percentage):
    """
    Plots a vertical side-by-side bar chart for keyword percentages,
    sorted descending by target label (Label 1 if present),
    with numeric values on top of each bar.
    """
    # Sort by "Label 1" if it exists, else by last column
    target_label = "Label 1" if "Label 1" in keyword_percentage.columns else keyword_percentage.columns[-1]
    keyword_percentage_sorted = keyword_percentage.sort_values(target_label, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot side-by-side bars: each label is a column, each keyword is an x-axis category
    keyword_percentage_sorted.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[COLOR_0, COLOR_1] if len(keyword_percentage_sorted.columns) == 2 else None,
        width=0.8
    )

    ax.set_xticklabels(keyword_percentage_sorted.index, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Keyword", fontsize=14)
    ax.set_ylabel("Proportion in Sentences (%)", fontsize=14)
    ax.set_title("Keyword Distribution by Label (Percentage)", fontsize=16)
    ax.legend(fontsize=12)

    # Annotate each bar on top
    for container in ax.containers:
        for bar in container.patches:
            height = bar.get_height()
            if height > 0:
                # x: center of bar, y: top of bar
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height
                ax.text(
                    x,
                    y + 0.2,            # 0.2 offset so label is above the bar
                    f"{height:.2f}%",   # 2 decimals + '%'
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=12
                )

    plt.tight_layout()
    plt.savefig("figures/keyword_percentage_vertical.png")
    plt.show()


# ------------------------------
# 7. Analyze Country Code Distribution
# ------------------------------
def analyze_country_code(df):
    # 1. Group data by country_code and label to get raw counts
    country_counts = df.groupby(["country_code", "label"]).size().reset_index(name="count")
    print("\nCountry code counts (head):")
    print(country_counts.head())

    # 2. Create pivot table (rows=country_code, columns=label)
    country_pivot = country_counts.pivot(index="country_code", columns="label", values="count").fillna(0)

    # 3. Dynamically rename columns (Label 0, Label 1, etc.)
    unique_labels = sorted(country_pivot.columns)
    country_pivot.columns = [f"Label {label}" for label in unique_labels]

    # 4. Calculate percentages for each country code
    label_totals = country_pivot.sum(axis=0)
    country_percentage = country_pivot.divide(label_totals, axis=1) * 100

    # 5. Sort descending by "Label 1" if it exists, otherwise by last column
    target_label = "Label 1" if "Label 1" in country_percentage.columns else country_percentage.columns[-1]
    country_percentage_sorted = country_percentage.sort_values(target_label, ascending=False)
    country_counts_sorted = country_pivot.loc[country_percentage_sorted.index]

    # 6. Plot vertical side-by-side bars for country code percentages
    plot_country_code_percentage_vertical(country_percentage_sorted)

    # 7. Plot vertical side-by-side bars for raw counts
    plot_country_code_counts_vertical(country_counts_sorted)


def plot_country_code_percentage_vertical(country_percentage_sorted):
    """
    Plots a vertical side-by-side bar chart for country code percentages,
    sorted descending by target label (Label 1 if present),
    with numeric values on top of each bar.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use DataFrame's built-in plotting to create side-by-side bars
    country_percentage_sorted.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[COLOR_0, COLOR_1] if len(country_percentage_sorted.columns) == 2 else None,
        width=0.8
    )

    ax.set_xticklabels(country_percentage_sorted.index, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Country Code", fontsize=14)
    ax.set_ylabel("Proportion in Sentences (%)", fontsize=14)
    ax.set_title("Country Code Distribution by Label (Percentage)", fontsize=16)
    ax.legend(fontsize=12)

    # Annotate each bar: place text slightly above the bar
    for container in ax.containers:
        for bar in container.patches:
            height = bar.get_height()
            if height > 0:
                # Center x at the midpoint of the bar, y slightly above the bar
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height
                # ax.text(
                #     x, 
                #     y + 0.2,           # 0.2 is an offset so text is above the bar
                #     f"{height:.2f}%",  # Format to 2 decimals plus '%'
                #     ha="center",
                #     va="bottom",
                #     color="black",     # Use black text if the top of the bar is not too high
                #     fontsize=12
                # )

    plt.tight_layout()
    plt.savefig("figures/country_code_percentage_vertical.png")
    plt.show()



def plot_country_code_counts_vertical(country_counts_sorted):
    """
    Plots a vertical side-by-side bar chart for country code raw counts,
    sorted in the same order as the percentage DataFrame.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Again, side-by-side bars for each label in columns
    country_counts_sorted.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[COLOR_0, COLOR_1] if len(country_counts_sorted.columns) == 2 else None,
        width=0.8
    )

    ax.set_xticklabels(country_counts_sorted.index, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Country Code", fontsize=14)
    ax.set_ylabel("Count of Sentences", fontsize=14)
    ax.set_title("Country Code Counts by Label (Vertical)", fontsize=16)
    ax.legend(fontsize=12)

    # Annotate each bar
    for container in ax.containers:
        for bar in container.patches:
            height = bar.get_height()
            if height > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2
                ax.text(x, y, f"{int(height)}", ha="center", va="center", color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig("figures/country_code_counts_vertical.png")
    plt.show()



# ------------------------------
# 8. Main Function
# ------------------------------
def main():
    # Load train data
    train_data = load_train_data()
    
    # Analyze label distribution and plot
    analyze_label_distribution(train_data)
    
    # Analyze word count and plot
    analyze_word_count(train_data)
    
    # Initialize RoBERTa tokenizer
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Analyze token count (using RoBERTa tokenizer) and plot
    analyze_token_count(train_data, roberta_tokenizer)
    
    # Analyze keyword counts and percentages, then plot
    analyze_keyword_counts(train_data)
    
    # Analyze country code distribution and plot
    analyze_country_code(train_data)
    
if __name__ == "__main__":
    main()
