import argparse
import pandas as pd


def update_prompt(x):
    y = x.copy()
    y[0] = {'content': 'You are a helpful assistant. Before providing your answer, think through the reasoning process internally. Your internal reasoning should be enclosed in <think></think> tags, and your final answer should be enclosed in <answer></answer> tags. For example:\n\n<think> Your reasoning process here </think>\n<answer> Your answer here </answer>\n\nNow, the user will present you with a competitive programming problem along with some sample input and output cases. You should verify your approach using these samples. Once you have determined a correct approach, provide the final python code implementation that can be directly submitted to an online judge. The code should use standard input/output and must not include any extra characters outside the <answer></answer> tags.', 'role': 'system'}
    return y
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_fname", type=str, default="")
    parser.add_argument("--out_fname", type=str, default="")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_fname)
    df_new = df.copy()
    df_new['prompt'] = df_new['prompt'].apply(update_prompt)
    # filter row i where len(df.iloc[i]['reward_model']['ground_truth']['input']) == 0
    df_new = df_new[df_new['reward_model'].apply(lambda x: len(x['ground_truth']['input']) > 0)]
    # shuffle df_new
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new.to_parquet(args.out_fname)
