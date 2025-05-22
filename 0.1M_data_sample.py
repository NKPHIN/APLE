import random
import csv # 用于处理CSV文件
# import json # 如果是json lines 文件

def line_by_line_probabilistic_sample(filepath, sample_ratio, has_header=True, delimiter=',', random_state=100):
    """
    逐行从文件中按概率随机采样数据。

    参数:
    filepath (str): 文件路径。
    sample_ratio (float): 要采样的比例 (例如, 0.1 表示 10%)。
    has_header (bool): 文件是否有表头。
    delimiter (str): CSV文件的分隔符。
    random_state (int, optional): 随机种子。

    返回:
    list: 包含采样行的列表 (如果是CSV，每行为一个列表；如果是纯文本，每行为一个字符串)。
           如果 has_header 为 True，则第一行为表头。
    """
    if not 0 < sample_ratio <= 1:
        raise ValueError("采样比例必须在 (0, 1] 之间")

    if random_state is not None:
        random.seed(random_state)

    sampled_lines = []
    header = None

    print(f"开始从文件 '{filepath}' 中逐行概率采样...")
    line_count = 0
    sampled_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # 根据你的文件编码调整 'utf-8'
            
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                try:
                    header = next(reader)
                    sampled_lines.append(header)
                    line_count +=1
                except StopIteration:
                    print("文件为空或只有表头。")
                    return [] if not has_header else [header] if header else []


            for row in reader:
                line_count += 1
                if random.random() < sample_ratio:
                    sampled_lines.append(row)
                    sampled_count += 1
                if line_count % 100000 == 0: # 每处理10万行打印一次进度
                    print(f"已处理 {line_count} 行，已采样 {sampled_count} 行...")

            
            

        print(f"采样完成。总共处理了 {line_count} 行，采样了 {sampled_count} 行。")
        if has_header and sampled_count == 0 and header: # 只采样了表头
             print("只采样到了表头，没有采样到数据行。")
        elif sampled_count == 0 and not header:
             print("没有采样到任何数据。")

        return sampled_lines

    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
        return []
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return []
    
# main函数，加载运行，保存
if __name__ == '__main__':
    filepath = 'Tenrec/ctr_data_1M.csv'
    sample_ratio = 0.1
    has_header = True
    delimiter = ','
    random_state = 100
    sampled_lines = line_by_line_probabilistic_sample(filepath, sample_ratio, has_header, delimiter, random_state)
    #先创建路径
    import os
    if not os.path.exists(f'Tenrec/ctr_data_1M/{str(sample_ratio*100)}'):
        os.makedirs(f'Tenrec/ctr_data_1M/{str(sample_ratio*100)}')
    with open(f'Tenrec/ctr_data_1M/{str(sample_ratio*100)}.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if has_header:
            writer.writerow(sampled_lines[0])
            sampled_lines = sampled_lines[1:]
        writer.writerows(sampled_lines)

