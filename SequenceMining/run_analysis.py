def spade_output(df, cv_data, id_data, url_data, support, maxlen, cv_pages):
    pip install pycspade
    import sys
    import pandas as pd
    from pycspade.helpers import spade, print_result
    import io
    import numpy as np

    #dfからspadeの結果を出力
    def get_raw_result(df,support,maxlen):
        #spadeが回らないのでソートする
        sorted_df2 = df.sort_values(by='cookie_index')
        #使用データの形に整形
        df_use = pd.DataFrame(sorted_df2, columns=["cookie_index","timestamp_index","url_index_x"])
        #リスト形式への変換
        data = df_use.apply(lambda row: [row['cookie_index'], row['timestamp_index'], [row['url_index_x']]], axis=1).tolist()
        #spade結果の出力、存在割合、パターン長さが引数
        result = spade(data=data, parse=True,support=support,maxlen=maxlen)

        return result
    
    result =  get_raw_result(df,support,maxlen)

    # pycspadeのprint_result関数の出力を文字列として取得するための関数
    def get_print_result_string(result):
        import sys
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print_result(result)
        return f.getvalue()

    result_string = get_print_result_string(result)

    #result_stringをきれいなtableに変換
    def get_result_table(result_string):
        # 文字列を行ごとに分割
        lines = result_string.strip().split('\n')

        # ヘッダー行をスキップ
        data = []
        for line in lines[1:]:
            parts = line.split()
            occurs = int(parts[0])
            accum = int(parts[1])
            support = float(parts[2])
            confid = parts[3] if parts[3] != 'N/A' else None
            lift = parts[4] if parts[4] != 'N/A' else None
            sequence = ' '.join(parts[5:])
            data.append([occurs, accum, support, confid, lift, sequence])

        # DataFrameに変換
        df_result = pd.DataFrame(data, columns=['Occurs', 'Accum', 'Support', 'Confid', 'Lift', 'Sequence'])

        #シーケンスの中身の形をデータの形に合わせる[33, 44]
        def convert_sequence(seq):
            # 矢印で区切られた部分をリストに変換
            return [int(item.strip('()')) for item in seq.split('->')]
        
        df_result['Sequence'] = df_result['Sequence'].apply(convert_sequence)

        return df_result
    
    df_result = get_result_table(result_string)

    def get_cv_table(df_result):
        #df_resultをcvページ含み、含まずに分類
        # cvページが含まれるかどうかを判定する関数
        def contains_cv_page(seq):
            return any(page in cv_pages for page in seq)

        # cvページが含まれる列をフィルタリング
        df_cv = df_result[df_result['Sequence'].apply(contains_cv_page)]
        # cvページが含まれない列をフィルタリング
        df_not_cv = df_result[~df_result['Sequence'].apply(contains_cv_page)]

        return df_cv, df_not_cv
    
    df_cv, df_not_cv = get_cv_table(df_result)

    #長さ2以上、Top50の抽出関数
    def get_seq_len2_top50(df_result):
        # Sequenceの要素数が2以上のレコードをフィルタリング
        filtered_df = df_result[df_result['Sequence'].apply(len) >= 2]
        # Occursの発生回数が多い順にソート
        sorted_df = filtered_df.sort_values(by='Occurs', ascending=False)
        # 上位100のレコードを取得
        top_50_df = sorted_df.head(50)
        # Sequenceのデータ
        sequences = top_50_df['Sequence'].tolist()
        return sequences
    
    sequences = get_seq_len2_top50(df_cv)
    sequences1 = get_seq_len2_top50(df_not_cv)

    def get_user_pattern_table(df, sequences):
        # 各Sequenceに対してカラムを追加し、0で初期化
        for seq in sequences:
            seq_str = ','.join(map(str, seq))
            df[seq_str] = 0
        # 各cookie_indexごとに処理
        results = []

        for cid, group in df.groupby('cookie_index'):
            # timestamp_indexの昇順に並べ替え
            group = group.sort_values(by='timestamp_index', ascending=True)
            act_nums = group['url_index_x'].tolist()
            result = {"cookie_index": cid}
            for seq in sequences:
                seq_str = ','.join(map(str, seq))
                seq_len = len(seq)
                found = 0
                seq_idx = 0
                for act in act_nums:
                    if act == seq[seq_idx]:
                        seq_idx += 1
                        if seq_idx == seq_len:
                            found = 1
                            break
                result[seq_str] = found
            results.append(result)

        # 結果のデータフレームを作成
        result_df = pd.DataFrame(results)
        return result_df

    result_df = get_user_pattern_table(df, sequences)
    result_df1 = get_user_pattern_table(df, sequences1)

    def join_result_cv(result_df, cv_data, id_data):
        result_df['cookie_index'] = result_df['cookie_index'].astype(str)
        id_data['cookie_index'] = id_data['cookie_index'].astype(str)

        # cv_dataとid_dataを結合する
        cv_data_join = pd.merge(cv_data, id_data, on='cookie_id', how='left')
        #ソートする(昇順の最後を採用、そのidがどこかでcvしていたらcv判定される)
        cv_data_join2 = cv_data_join.sort_values('default_cv_flg').drop_duplicates(['cookie_index'], keep='last')

        # result_dfにcv_data_joinのcookie_idとdefault_cv_flgを結合する
        result_df_join = pd.merge(result_df, cv_data_join2[['cookie_index', 'default_cv_flg']], on='cookie_index',how='left')
        
        return result_df_join
    
    result_df_join_pd = join_result_cv(result_df, cv_data, id_data)
    result_df_join_pd1 = join_result_cv(result_df1, cv_data, id_data)

    def get_corr_ratio_table(result_df_join_pd):
        ##相関の結果
        correlation_matrix = result_df_join_pd.corr()
        correlation_with_cv = pd.DataFrame({
            'index': correlation_matrix.index,
            'correlation_with_default_cv_flg': correlation_matrix['default_cv_flg']
        }).reset_index(drop=True)
        # cookie_indexの行とdefault_cv_flgの行を削除
        correlation_with_cv = correlation_with_cv[
            (correlation_with_cv['index'] != 'cookie_index') & 
            (correlation_with_cv['index'] != 'default_cv_flg')
        ]

        ##CV率の計算
        average_cv_flg = []
        columns_list = []
        # 各列についてフラグが立っている行のdefault_cv_flgの平均を計算
        for column in result_df_join_pd.columns:
            if column not in ['cookie_index', 'default_cv_flg']:
                # フラグが立っている行を抽出
                flagged_rows = result_df_join_pd[result_df_join_pd[column] == 1]
                # default_cv_flgの平均を計算
                mean_cv_flg = flagged_rows['default_cv_flg'].mean()
                # 結果を格納
                average_cv_flg.append(mean_cv_flg)
                columns_list.append(column)
        # 結果を新しいデータフレームに格納
        average_cv_flg_df = pd.DataFrame({
            'column': columns_list,
            'average_cv_flg': average_cv_flg
        })

        # データフレームの結合
        merged_df = pd.merge(correlation_with_cv, average_cv_flg_df, left_on='index', right_on='column', how='inner').drop(columns=['column'])

        return merged_df
    
    #support情報の追加、result_df/result_df1を参照して作成
    def join_support(result_df,merged_df):
        #result_dfからcookie_indexをのぞいた各列について、合計/列数を行う
        #Drop 'cookie_index' column
        result_df_drop_index = result_df.drop(columns=['cookie_index'])
        #Calculate sum and count for each column
        sum_series = result_df_drop_index.sum()
        count_series = result_df_drop_index.count()
        #Create a new DataFrame with the results
        summary_df = pd.DataFrame({
            'index': sum_series.index,
            'appearance rate': sum_series.values / count_series.values
        })

        #テーブルデータをmerged_dfに結合する
        merged_df3 = pd.merge(merged_df, summary_df, on='index', how='left')

        return merged_df3

    # URLカラムの作成
    def join_url(url_data,merged_df):
        # URLリストの辞書を作成
        url_dict = dict(zip(url_data['url_index'].astype(str), url_data['url_3']))
        merged_df['url_index'] = merged_df['index'].apply(lambda x: ','.join([url_dict[i] for i in x.split(',')]))
        return merged_df

    merged_df = get_corr_ratio_table(result_df_join_pd)
    merged_df1 = get_corr_ratio_table(result_df_join_pd1)

    merged_df = join_support(result_df,merged_df)
    merged_df1 = join_support(result_df1,merged_df1)

    merged_df = join_url(url_data,merged_df)
    merged_df1 = join_url(url_data,merged_df1)

    return merged_df, merged_df1