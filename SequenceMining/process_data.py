from pyspark.sql import SparkSession
import pandas as pd
from IPython.display import display
import numpy as np
from IPython.display import Image, display
import os
from datetime import datetime
import pytz
import re

# 特定のカラムを抽出
def extract_columns(df_pandas, columns_to_extract):
    return df_pandas[columns_to_extract]

# 各セッションIDごとに最も早いタイムスタンプを持つ1行を取得
def get_first_session_per_id(df_selected):
    return df_selected.loc[df_selected.groupby([COOKIE_ID, SESSION_ID])[TIME_STAMP].idxmin()].drop_duplicates()

# 新規フラグの設定
def set_new_flag(distinct_df, new_visit_days=7):
    distinct_df = distinct_df.sort_values(by=[COOKIE_ID, SESSION_ID, TIME_STAMP])
    distinct_df[TIME_STAMP]=pd.to_datetime(distinct_df[TIME_STAMP])
    distinct_df['prev_date'] = distinct_df.groupby(COOKIE_ID)[TIME_STAMP].shift(1)
    distinct_df['days_diff'] = (distinct_df[TIME_STAMP] - distinct_df['prev_date']).dt.days
    distinct_df['new_flag'] = 0
    distinct_df.loc[distinct_df['prev_date'].isna() | (distinct_df['days_diff'] >= new_visit_days), 'new_flag'] = 1
    return distinct_df

# 新規フラグに基づいた新クッキーIDの割り振り
def create_and_save_cookie_id_mapping(distinct_df):
    distinct_df['cumulative_sum'] = distinct_df.groupby(COOKIE_ID)['new_flag'].cumsum()
    distinct_df['new_cookie_id'] = distinct_df[COOKIE_ID] + '_' + distinct_df['cumulative_sum'].astype(str)
    new_cookie_tmp = distinct_df[[COOKIE_ID, SESSION_ID, 'new_cookie_id']]
    return new_cookie_tmp

# リピート回数の計算
def calculate_repeat_count(distinct_df):
    def count_repeats(group):
        count = 0
        repeat_counts = []
        for flag in group['new_flag']:
            if flag == 0:
                repeat_counts.append(count)
                count += 1
            else:
                repeat_counts.append(0)
                count = 1
        return repeat_counts

    distinct_df['repeat_count'] = distinct_df.groupby('new_cookie_id').apply(lambda group: count_repeats(group)).explode().values
    return distinct_df

# 累積訪問回数の計算
def calculate_visit_count(distinct_df):
    distinct_df['visit_cnt'] = distinct_df.sort_values(by=['new_cookie_id', TIME_STAMP]).groupby('new_cookie_id').cumcount() + 1
    return distinct_df

# データの結合
def merge_data(df_selected, distinct_df):
    return pd.merge(df_selected, distinct_df[['new_cookie_id', SESSION_ID, 'new_flag', 'repeat_count', 'visit_cnt']], on=SESSION_ID, how='left')

# 各url_pageの出現回数と出現率を計算
def calculate_url_page_ratios(df_selected):
    url_page_counts = df_selected[URL].value_counts()
    total_records = len(df_selected)
    url_page_ratios = ((url_page_counts / total_records) * 100).round(2)
    df_ratios = url_page_ratios.reset_index()
    df_ratios.columns = [URL, 'appearance_rate']
    return df_ratios

# 出現率のカラムを追加
def add_appearance_rate(df_unique, df_ratios):
    df_unique = df_unique.merge(df_ratios, on=URL, how='left')
    df_unique['appearance_rate'].fillna(0, inplace=True)
    df_unique['appearance_rate'] = df_unique['appearance_rate'].astype(float)
    return df_unique

# url_pageのユニークな数を取得
def get_unique_url_page_count(df_selected):
    return df_selected[URL].nunique()

# 絞り込み方法を選択し、url_pageの一覧を取得、インデックスを付与
def select_and_process_data(filter_option, df_top_percentage, df_filtered_by_rate, df_filtered_by_cumsum):
    if filter_option == 1:
        df_selected2 = df_top_percentage
    elif filter_option == 2:
        df_selected2 = df_filtered_by_rate
    elif filter_option == 3:
        df_selected2 = df_filtered_by_cumsum
    else:
        df_selected2 = pd.DataFrame()

    df_selected2 = df_selected2.reset_index(drop=True)
    df_selected2['index'] = df_selected2.index

    if URL in df_selected2.columns:
        return df_selected2[URL].unique().tolist()
    else:
        print(f"{URL}列が存在しません。")
        return []

# url_pageの一覧を表示
def display_url_page_list(filter_option, df_top_percentage, df_filtered_by_rate, df_filtered_by_cumsum):
    url_page_list = select_and_process_data(filter_option, df_top_percentage, df_filtered_by_rate, df_filtered_by_cumsum)
    print("\n抽出したURL一覧:")
    for index, value in enumerate(url_page_list):
        print(f"{index}: {value}")
    return url_page_list

# 使用するurl_pageの分析
def analyze_url_page(df_unique, df_ratios, percentage, threshold, threshold_percentage, filter_option):
    df_unique_sorted = df_unique.sort_values(by='appearance_rate', ascending=False)
    top_percentage_count = int(len(df_unique_sorted) * percentage)
    df_top_percentage = df_unique_sorted.head(top_percentage_count)
    df_filtered_by_rate = df_unique_sorted[df_unique_sorted['appearance_rate'] >= threshold]
    df_unique_sorted['cumulative_ratio'] = df_unique_sorted['appearance_rate'].cumsum()
    df_filtered_by_cumsum = df_unique_sorted[df_unique_sorted['cumulative_ratio'] <= threshold_percentage]

    print(f"全URLの数: {df_unique[URL].nunique()}")
    print(f"\n各絞り込み後URL数\n①上位{percentage * 100}%のユニークなURL数: {df_top_percentage[URL].nunique()}")
    print(f"②登場率が{threshold}%以上のURL数: {df_filtered_by_rate[URL].nunique()}")
    print(f"③累積構成比が{threshold_percentage}%を超えるまでのURL数: {df_filtered_by_cumsum.shape[0]}")

    return display_url_page_list(filter_option, df_top_percentage, df_filtered_by_rate, df_filtered_by_cumsum)

# カラム名を設定する関数
def set_column_names(columns):
    global COOKIE_ID, SESSION_ID, TIME_STAMP, URL, DEFAULT_CV_FLG
    COOKIE_ID = columns["cookie_id"]
    SESSION_ID = columns["session_id"]
    TIME_STAMP = columns["time_stamp"]
    URL = columns["url"]
    DEFAULT_CV_FLG = columns["default_cv_flg"]

# メイン関数
def main_option(df_pandas, new_visit_days, filter_option, percentage, threshold, threshold_percentage, columns):
    global df_selected, url_page_list
    set_column_names(columns)
    columns_to_extract = [COOKIE_ID, SESSION_ID, URL, TIME_STAMP, DEFAULT_CV_FLG]
    df_selected = extract_columns(df_pandas, columns_to_extract)
    distinct_df = get_first_session_per_id(df_selected)
    distinct_df = set_new_flag(distinct_df, new_visit_days)
    result_df = create_and_save_cookie_id_mapping(distinct_df)
    distinct_df = calculate_repeat_count(distinct_df)
    distinct_df = calculate_visit_count(distinct_df)
    df_selected = merge_data(df_selected, distinct_df)
    df_ratios = calculate_url_page_ratios(df_selected)
    columns_to_extract_unique = [URL]
    df_unique = df_selected[columns_to_extract_unique].drop_duplicates(subset=[URL])
    df_unique = add_appearance_rate(df_unique, df_ratios)
    unique_url_page_count = get_unique_url_page_count(df_selected)
    url_page_list = analyze_url_page(df_unique, df_ratios, percentage, threshold, threshold_percentage, filter_option)
    return df_selected, url_page_list


def process_and_save_data(df_selected, url_page_list, cv_url_page_index, columns):
    # CVとするURLを一つ選択し、URLの左の数字を[]の中に記載
    cv_url_page = url_page_list[cv_url_page_index]

    # 元データからurl_pageを絞り込む
    df_selected_filtered = df_selected[df_selected[columns['url']].isin(url_page_list)].copy()

    # url_pageに対応する先ほど抽出したpost_indexを取得
    post_index_dict = pd.DataFrame(url_page_list, columns=[columns['url']]).reset_index().set_index(columns['url'])['index'].to_dict()

    # cv_flgとpost_indexを追加
    df_selected_filtered.loc[:, 'cv_flg'] = df_selected_filtered[columns['url']].apply(lambda x: 1 if x == cv_url_page else 0)
    df_selected_filtered.loc[:, 'url_index'] = df_selected_filtered[columns['url']].map(post_index_dict)

    # 以下、process_data関数の処理を実行
    # URLとpost_indexのユニークな組み合わせを抽出
    unique_combinations = df_selected_filtered[[URL, 'url_index']].drop_duplicates()

    # post_indexの昇順にソート
    unique_combinations_sorted = unique_combinations.sort_values(by='url_index')

    # ユニークなCOOKIE_IDにインデックスを付ける
    unique_visid = df_selected_filtered[COOKIE_ID].unique()
    visid_to_index = {visid: idx for idx, visid in enumerate(unique_visid)}

    # visid_to_indexをデータフレームに変換
    visid_index_df = pd.DataFrame(list(visid_to_index.items()), columns=[COOKIE_ID, 'cookie_index'])

    # ユニークなSESSION_IDにインデックスを付ける
    unique_session = df_selected_filtered[SESSION_ID].unique()
    session_to_index = {session: idx for idx, session in enumerate(unique_session)}

    # session_to_indexをデータフレームに変換
    session_to_index_df = pd.DataFrame(list(session_to_index.items()), columns=[SESSION_ID, 'session_index'])

    # 必要なカラムの抽出
    df_visits = df_selected_filtered[[COOKIE_ID, SESSION_ID, TIME_STAMP, URL, 'url_index']]
    df_url_groups = unique_combinations_sorted[[URL, 'url_index']]
    df_cookieid = visid_index_df[[COOKIE_ID, 'cookie_index']]
    df_session = session_to_index_df[[SESSION_ID, 'session_index']]

    # URLとURLグループの対応表をマージ
    #df_merged = pd.merge(df_visits, df_url_groups, on=URL, how='left')

    # クッキーIDと対応表をマージ
    df_merged = pd.merge(df_visits, df_cookieid, on=COOKIE_ID, how='left')

    # セッションIDと対応表をマージ
    df_merged = pd.merge(df_merged, df_session, on=SESSION_ID, how='left')

    # クッキーIDごとにセッションIDをタイムスタンプ順に並べる
    df_result = df_merged.sort_values(by=[COOKIE_ID, TIME_STAMP])

    # クッキーIDごとにタイムスタンプの早い順にインデックスを付ける
    df_result['timestamp_index'] = df_result.groupby('cookie_index').cumcount() + 1

    # 必要なカラムの抽出
    df_final = df_result[['cookie_index', 'session_index', 'timestamp_index', 'url_index']]

    return unique_combinations_sorted, visid_index_df, session_to_index_df, df_final
