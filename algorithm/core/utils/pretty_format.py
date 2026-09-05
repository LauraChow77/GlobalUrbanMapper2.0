def join_str(joiner="-", content_list=[]):
    """
    使用joiner连接content_list内的所有内容
    :param joiner: 连接符，默认为"-"
    :param content_list: 待连接的内容list，每个元素需为str
    :return: 用joiner连接好的content_list
    """
    return joiner.join(content_list)


def organize_info(key=[], value=[]):
    """
    重新组织信息成以下格式：
    [** {key} **]: {value}
    :param key: 索引名
    :param value: 每个索引对应的值
    :return: 组织好的str
    """
    assert len(key) == len(value)

    info = []
    length = len(key)
    for idx in range(length):
        info.append("[** " + key[idx] + " **]: " + str(value[idx]))

    return join_str("\n", info)
