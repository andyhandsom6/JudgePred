import pandas as pd
import torch
import json
import math
import pandas.api.types

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass

accusation_to_id_map = {"伪证罪": 0, "假冒专利罪": 1, "故意泄露国家秘密罪": 2, "运送他人偷越国（边）境罪": 3, "破坏计算机信息系统罪": 4, "伪造、变造、买卖武装部队公文、证件、印章罪": 5, "伪造货币罪": 6, "聚众冲击军事禁区罪": 7, "刑讯逼供罪": 8, "违规披露、不披露重要信息罪": 9, "扰乱法庭秩序罪": 10, "过失以危险方法危害公共安全罪": 11, "利用未公开信息交易罪": 12, "非法生产、买卖、运输制毒物品、走私制毒物品罪": 13, "虐待被监管人罪": 14, "非法携带枪支、弹药、管制刀具、危险物品危及公共安全罪": 15, "侮辱罪": 16, "组织淫秽表演罪": 17, "非法经营同类营业罪": 18, "侵犯公民个人信息罪": 19, "袭警罪": 20, "破坏电力设备罪": 21, "扰乱无线电通讯管理秩序罪": 22, "非法利用信息网络罪": 23, "冒充军人招摇撞骗罪": 24, "金融凭证诈骗罪": 25, "串通投标罪": 26, "私分罚没财物罪": 27, "损害商业信誉、商品声誉罪": 28, "集资诈骗罪": 29, "非法购买增值税专用发票、购买伪造的增值税专用发票罪": 30, "妨害作证罪": 31, "引诱、教唆、欺骗他人吸毒罪": 32, "非法生产、买卖警用装备罪": 33, "内幕交易、泄露内幕信息罪": 34, "保险诈骗罪": 35, "为亲友非法牟利罪": 36, "行贿罪": 37, "虚开发票罪": 38, "销售侵权复制品罪": 39, "消防责任事故罪": 40, "伪造、变造、买卖身份证件罪": 41, "寻衅滋事罪": 42, "组织、利用会道门、邪教组织、利用迷信致人重伤、死亡罪": 43, "走私贵重金属罪": 44, "盗掘古人类化石、古脊椎动物化石罪": 45, "交通肇事罪": 46, "非法批准征收、征用、占用土地罪": 47, "伪造、变造金融票证罪": 48, "窃取、收买、非法提供信用卡信息罪": 49, "聚众斗殴罪": 50, "破坏易燃易爆设备罪": 51, "过失损坏武器装备、军事设施、军事通信罪": 52, "组织、资助非法聚集罪": 53, "非法获取国家秘密罪": 54, "伪造、变造、买卖国家机关公文、证件、印章罪": 55, "编造、故意传播虚假信息罪": 56, "帮助毁灭、伪造证据罪": 57, "贪污罪": 58, "侵犯著作权罪": 59, "伪造公司、企业、事业单位、人民团体印章罪": 60, "生产、销售不符合卫生标准的化妆品罪": 61, "滥用职权罪": 62, "妨害动植物防疫、检疫罪": 63, "非法出售发票罪": 64, "签订、履行合同失职被骗罪": 65, "虚假诉讼罪": 66, "倒卖文物罪": 67, "盗窃、侮辱、故意毁坏尸体、尸骨、骨灰罪": 68, "利用影响力受贿罪": 69, "违法发放林木采伐许可证罪": 70, "私放在押人员罪": 71, "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪": 72, "受贿罪": 73, "提供伪造、变造的出入境证件罪": 74, "失职造成珍贵文物损毁、流失罪": 75, "组织残疾人、儿童乞讨罪": 76, "生产、销售伪劣农药、兽药、化肥、种子罪": 77, "猥亵儿童罪": 78, "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪": 79, "非法制造、出售非法制造的用于骗取出口退税、抵扣税款发票罪": 80, "生产、销售伪劣产品罪": 81, "盗伐林木罪": 82, "伪造、盗窃、买卖、非法提供、非法使用武装部队专用标志罪": 83, "拒不支付劳动报酬罪": 84, "暴力取证罪": 85, "逃避追缴欠税罪": 86, "非法侵入住宅罪": 87, "私分国有资产罪": 88, "赌博罪": 89, "单位受贿罪": 90, "非法持有、私藏枪支、弹药罪": 91, "帮助信息网络犯罪活动罪": 92, "走私国家禁止进出口的货物、物品罪": 93, "玩忽职守罪": 94, "绑架罪": 95, "不报、谎报安全事故罪": 96, "逃避商检罪": 97, "协助组织卖淫罪": 98, "聚众哄抢罪": 99, "强迫交易罪": 100, "非法使用窃听、窃照专用器材罪": 101, "强迫卖淫罪": 102, "徇私舞弊减刑、假释、暂予监外执行罪": 103, "破坏选举罪": 104, "生产、销售不符合安全标准的产品罪": 105, "非法经营罪": 106, "失职致使在押人员脱逃罪": 107, "破坏交通设施罪": 108, "非法进行节育手术罪": 109, "走私废物罪": 110, "失火罪": 111, "动植物检疫徇私舞弊罪": 112, "劫夺被押解人员罪": 113, "包庇毒品犯罪分子罪": 114, "吸收客户资金不入账罪": 115, "走私珍贵动物、珍贵动物制品罪": 116, "破坏武器装备、军事设施、军事通信罪": 117, "盗窃、抢夺、毁灭国家机关公文、证件、印章罪": 118, "引诱幼女卖淫罪": 119, "非法猎捕、收购、运输、出售陆生野生动物罪": 120, "脱逃罪": 121, "徇私舞弊不移交刑事案件罪": 122, "组织、领导传销活动罪": 123, "盗掘古文化遗址、古墓葬罪": 124, "挪用特定款物罪": 125, "扰乱国家机关工作秩序罪": 126, "非法出售用于骗取出口退税、抵扣税款发票罪": 127, "擅自发行股票、公司、企业债券罪": 128, "生产、销售不符合标准的医用器材罪": 129, "侵犯商业秘密罪": 130, "包庇、纵容黑社会性质组织罪": 131, "过失损坏广播电视设施、公用电信设施罪": 132, "民事、行政枉法裁判罪": 133, "代替考试罪": 134, "投放危险物质罪": 135, "打击报复证人罪": 136, "编造、故意传播虚假恐怖信息罪": 137, "销售假冒注册商标的商品罪": 138, "收买被拐卖的妇女、儿童罪": 139, "故意损毁文物罪": 140, "虚假广告罪": 141, "非法狩猎罪": 142, "走私、贩卖、运输、制造毒品罪": 143, "信用卡诈骗罪": 144, "欺诈发行证券罪": 145, "假冒注册商标罪": 146, "危害珍贵、濒危野生动物罪": 147, "妨害清算罪": 148, "对非国家工作人员行贿罪": 149, "强令、组织他人违章冒险作业罪": 150, "非法收购、运输盗伐、滥伐的林木罪": 151, "单位行贿罪": 152, "传授犯罪方法罪": 153, "非法制造、出售非法制造的发票罪": 154, "票据诈骗罪": 155, "滥伐林木罪": 156, "组织未成年人进行违反治安管理活动罪": 157, "盗窃罪": 158, "非法捕捞水产品罪": 159, "对单位行贿罪": 160, "劫持船只、汽车罪": 161, "工程重大安全事故罪": 162, "骗购外汇罪": 163, "帮助犯罪分子逃避处罚罪": 164, "抢劫罪": 165, "伪造、出售伪造的增值税专用发票罪": 166, "敲诈勒索罪": 167, "非法出售增值税专用发票罪": 168, "遗弃罪": 169, "拐卖妇女、儿童罪": 170, "放火罪": 171, "过失损毁文物罪": 172, "催收非法债务罪": 173, "非法行医罪": 174, "危险物品肇事罪": 175, "生产、销售不符合安全标准的食品罪": 176, "拒不执行判决、裁定罪": 177, "出售、购买、运输假币罪": 178, "虐待罪": 179, "虚假破产罪": 180, "过失致人死亡罪": 181, "辩护人、诉讼代理人毁灭证据、伪造证据、妨害作证罪": 182, "传播淫秽物品罪": 183, "生产、销售有毒、有害食品罪": 184, "爆炸罪": 185, "破坏监管秩序罪": 186, "妨害传染病防治罪": 187, "制作、复制、出版、贩卖、传播淫秽物品牟利罪": 188, "强迫他人吸毒罪": 189, "国有公司、企业、事业单位人员失职罪": 190, "非法处置查封、扣押、冻结的财产罪": 191, "非法获取计算机信息系统数据、非法控制计算机信息系统罪": 192, "挪用公款罪": 193, "妨害公务罪": 194, "环境监管失职罪": 195, "传播性病罪": 196, "职务侵占罪": 197, "掩饰、隐瞒犯罪所得、犯罪所得收益罪": 198, "非法种植毒品原植物罪": 199, "过失投放危险物质罪": 200, "徇私枉法罪": 201, "倒卖车票、船票罪": 202, "非法集会、游行、示威罪": 203, "危险驾驶罪": 204, "宣扬恐怖主义、极端主义、煽动实施恐怖活动罪": 205, "骗取出境证件罪": 206, "引诱、容留、介绍卖淫罪": 207, "逃汇罪": 208, "出售出入境证件罪": 209, "组织考试作弊罪": 210, "出具证明文件重大失实罪": 211, "非法侵入计算机信息系统罪": 212, "组织、领导、参加黑社会性质组织罪": 213, "抢夺罪": 214, "走私普通货物、物品罪": 215, "高利转贷罪": 216, "非法生产、销售专用间谍器材、窃听、窃照专用器材罪": 217, "重婚罪": 218, "洗钱罪": 219, "挪用资金罪": 220, "非法采矿罪": 221, "开设赌场罪": 222, "破坏生产经营罪": 223, "强奸罪": 224, "提供侵入、非法控制计算机信息系统程序、工具罪": 225, "使用虚假身份证件、盗用身份证件罪": 226, "持有、使用假币罪": 227, "重大责任事故罪": 228, "侮辱国旗、国徽、国歌罪": 229, "非国家工作人员受贿罪": 230, "投放虚假危险物质罪": 231, "非法生产、买卖武装部队制式服装罪": 232, "过失损坏易燃易爆设备罪": 233, "非法搜查罪": 234, "食品、药品监管渎职罪": 235, "信用证诈骗罪": 236, "污染环境罪": 237, "诽谤罪": 238, "执行判决、裁定滥用职权罪": 239, "违法发放贷款罪": 240, "破坏交通工具罪": 241, "聚众扰乱社会秩序罪": 242, "窝藏、包庇罪": 243, "过失损坏电力设备罪": 244, "组织他人偷越国（边）境罪": 245, "容留他人吸毒罪": 246, "操纵证券、期货市场罪": 247, "提供虚假证明文件罪": 248, "非法处置进口的固体废物罪": 249, "强制猥亵、侮辱罪": 250, "非法制造、销售非法制造的注册商标标识罪": 251, "强迫劳动罪": 252, "故意损毁名胜古迹罪": 253, "大型群众性活动重大安全事故罪": 254, "违规出具金融票证罪": 255, "组织出卖人体器官罪": 256, "逃税罪": 257, "招收公务员、学生徇私舞弊罪": 258, "非法持有毒品罪": 259, "贷款诈骗罪": 260, "教育设施重大安全事故罪": 261, "组织卖淫罪": 262, "徇私舞弊不征、少征税款罪": 263, "合同诈骗罪": 264, "伪造、倒卖伪造的有价票证罪": 265, "阻碍军人执行职务罪": 266, "介绍贿赂罪": 267, "非法吸收公众存款罪": 268, "组织、利用会道门、邪教组织、利用迷信破坏法律实施罪": 269, "有价证券诈骗罪": 270, "非法出租、出借枪支罪": 271, "危害国家重点保护植物罪": 272, "聚众淫乱罪": 273, "虚报注册资本罪": 274, "骗取贷款、票据承兑、金融票证罪": 275, "破坏广播电视设施、公用电信设施罪": 276, "虐待被监护、看护人罪": 277, "伪造、变造国家有价证券罪": 278, "巨额财产来源不明罪": 279, "持有伪造的发票罪": 280, "故意毁坏财物罪": 281, "招摇撞骗罪": 282, "妨害信用卡管理罪": 283, "虚假出资、抽逃出资罪": 284, "拐骗儿童罪": 285, "诈骗罪": 286, "医疗事故罪": 287, "过失爆炸罪": 288, "组织越狱罪": 289, "破坏性采矿罪": 290, "非法拘禁罪": 291, "生产、销售、提供假药罪": 292, "以危险方法危害公共安全罪": 293, "走私武器、弹药罪": 294, "过失致人重伤罪": 295, "偷越国（边）境罪": 296, "故意伤害罪": 297, "非法组织卖血罪": 298, "骗取出口退税罪": 299, "侵占罪": 300, "聚众扰乱公共场所秩序、交通秩序罪": 301, "放纵走私罪": 302, "铁路运营安全事故罪": 303, "对有影响力的人行贿罪": 304, "过失决水罪": 305, "非法买卖、运输、携带、持有毒品原植物种子、幼苗罪": 306, "盗窃、抢夺枪支、弹药、爆炸物、危险物质罪": 307, "聚众冲击国家机关罪": 308, "重大劳动安全事故罪": 309, "非法占用农用地罪": 310, "非法制造、买卖、运输、储存危险物质罪": 311, "非法转让、倒卖土地使用权罪": 312, "故意杀人罪": 313, "侵犯通信自由罪": 314, "隐匿、故意销毁会计凭证、会计账簿、财务会计报告罪": 315, "窝藏、转移、隐瞒毒品、毒赃罪": 316, "诬告陷害罪": 317, "非法出售、提供试题、答案罪": 318, "非法低价出让国有土地使用权罪": 319, "国有公司、企业、事业单位人员滥用职权罪": 320}

def get_weight(defendant_num):
    """
        给定案件被告人人数，计算基础权重
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        float: 权重值，计算方式为log2(D), D为被告人数量
    """
    if isinstance(defendant_num, torch.Tensor):
        # defendant_num为tensor类型
        return math.log(defendant_num.item(), 2)
    else:
        # defendant_num为直接的数值类型
        return math.log(defendant_num, 2)

def get_per_defendant_metrics_weight(defendant_num):
    """
        计算最终结果中每个被告人对应的权重，是get_weight(defendant_num)/defendant_num
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        _type_: _description_
    """
    if isinstance(defendant_num, torch.Tensor):
        assert defendant_num.item() != 0
        return get_weight(defendant_num.item()) / defendant_num.item()
    else:
        assert defendant_num != 0
        return get_weight(defendant_num) / defendant_num

def get_weighted_per_defendant_metrics(outputs, labels, weight_tensor):
    """
        通过计算per-defendant指标计算per-case指标后进行加权
    Args:
        outputs (_type_): 模型预测
        labels (_type_): 真实标签
        weight_tensor (_type_): 这里的每个weight为get_weight(defendant_num)/defendant_num
    Returns:
        acc, precision, recall, f1: 加权后的最终指标
    """
    assert outputs.shape == labels.shape
    assert outputs.size(0) == weight_tensor.size(0)
    acc, precision, recall, f1 = 0, 0, 0, 0
    sample_num = outputs.size(0)
    # 加权平均的分母
    weight_total = torch.sum(weight_tensor).item()
    TPs = torch.sum(outputs * labels, dim=1)
    FPs = torch.sum(outputs * (1 - labels), dim=1)
    FNs = torch.sum((1 - outputs) * labels, dim=1)
    TNs = torch.sum((1 - outputs) * (1 - labels), dim=1)
    precisions = TPs / torch.clamp(TPs + FPs, min=1e-8)
    recalls = TPs / torch.clamp(TPs + FNs, min=1e-8)
    f1s = 2 * precisions * recalls / torch.clamp(precisions + recalls, min=1e-8)
    # exact match accuracy
    acc = torch.sum(weight_tensor * torch.all(outputs == labels, dim=1).float()) / weight_total
    precision = torch.sum(weight_tensor * precisions) / weight_total
    recall = torch.sum(weight_tensor * recalls) / weight_total
    f1 = torch.sum(weight_tensor * f1s) / weight_total
    return acc.item(), precision.item(), recall.item(), f1.item()

def get_case_level_metrics_by_per_defendant_metrics(outputs, labels, defendant_nums):
    """
        计算case-level metrics
    Args:
        outputs (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        labels (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        defendant_nums (List): len(batch_size) 每条输入对应案件的被告人数量数组
    """
    # 对每个被告人计算accuracy, precision, recall, f1
    weight_tensor = torch.tensor(list(map(get_per_defendant_metrics_weight, defendant_nums))).to(outputs.device)
    charge_acc, charge_precision, charge_recall, charge_f1 = get_weighted_per_defendant_metrics(outputs, labels, weight_tensor)
    result = {
        "case_level_Acc_Acc": charge_acc,
        "case_level_Acc_Precision": charge_precision,
        "case_level_Acc_Recall": charge_recall,
        "case_level_Acc_F1": charge_f1,
    }
    # print(result)
    return result

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    计算案件级别的F1分数，基于被告人级别的预测结果加权平均
    
    该函数将solution和submission中的罪名预测结果转换为one-hot编码，
    然后计算加权平均的case-level指标，最终返回F1分数。
    
    Args:
        solution (pd.DataFrame): 包含真实标签的数据框，必须有'id'和'gold_accusation'列
        submission (pd.DataFrame): 包含预测结果的数据框，必须有'id'和'accusations'列
        row_id_column_name (str): 用于匹配solution和submission的ID列名
        
    Returns:
        float: 加权平均后的case-level F1分数
        
    Note:
        - 每个被告人的所有罪名需要以分号分隔
        - 每个被告人的罪名需要以逗号分隔
        - 输入样例参考sample_submission.csv
        - 权重计算基于被告人数量，使用log2(D)/D的公式
        - 最终分数是case-level的加权平均F1分数
    '''
    # 加载罪名到ID的映射
    # with open('data/big/charge2idx_big.json', 'r', encoding='utf-8') as f:
    #     accusation_to_id_map = json.load(f)
    
    # 处理solution和submission中的罪名列表
    solution['gold_accusation'] = solution['gold_accusation'].str.split(';').apply(
        lambda x: [a.split(',') for a in x] if isinstance(x, list) else []
    )
    submission['accusations'] = submission['accusations'].str.split(';').apply(
        lambda x: [a.split(',') for a in x] if isinstance(x, list) else []
    )
    
    # 转换为one-hot编码
    flat_predictions = []
    flat_labels = []
    defendant_counts = []
    
    for _, row in submission.iterrows():
        case_id = row['id']
        solution_row = solution[solution['id'] == case_id].iloc[0]
        
        # 获取每个被告人的罪名预测和真实标签
        pred_accs = row['accusations']
        gold_accs = solution_row['gold_accusation']
        
        num_defendants = len(pred_accs)
        defendant_counts.extend([num_defendants] * num_defendants)
        
        # 转换为one-hot
        for i in range(num_defendants):
            pred_one_hot = [0] * len(accusation_to_id_map)
            for acc in pred_accs[i]:
                if acc in accusation_to_id_map:
                    pred_one_hot[accusation_to_id_map[acc]] = 1
            
            gold_one_hot = [0] * len(accusation_to_id_map)
            for acc in gold_accs[i]:
                if acc in accusation_to_id_map:
                    gold_one_hot[accusation_to_id_map[acc]] = 1
            
            flat_predictions.append(pred_one_hot)
            flat_labels.append(gold_one_hot)
    
    # 转换为tensor
    predictions_tensor = torch.tensor(flat_predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(flat_labels, dtype=torch.float32)
    defendant_nums = torch.tensor(defendant_counts, dtype=torch.long)
    
    # 计算指标
    result = get_case_level_metrics_by_per_defendant_metrics(
        predictions_tensor,
        labels_tensor,
        defendant_nums
    )
    
    # 返回F1分数作为最终得分
    return result['case_level_Acc_F1']

if __name__ == '__main__':
    solution_data = {
        'id': ['1', '2', '3'],
        'gold_accusation': [
            "盗窃罪;故意伤害罪,非法拘禁罪",
            "诈骗罪",
            "抢劫罪,故意杀人罪;非法持有枪支罪"
        ]
    }
    submission_data = {
        'id': ['1', '2', '3'],
        'accusations': [
            "盗窃罪;故意伤害罪,非法拘禁罪",
            "诈骗罪,合同诈骗罪",
            "抢劫罪,故意杀人罪;非法持有枪支罪,非法制造枪支罪"
        ]
    }
    
    solution = pd.DataFrame(solution_data)
    submission = pd.DataFrame(submission_data)
    row_id_column_name = 'id'
    
    score_result = score(solution, submission, row_id_column_name)
    print(f"Case-level F1 score: {score_result}")

    # solution = df.loc[:, ['id', 'gold_accusation']].copy()
    # submission = df.loc[:, ['id', 'accusations']].copy()
    # row_id_column_name = 'id'
    # score(solution, submission, row_id_column_name)

    