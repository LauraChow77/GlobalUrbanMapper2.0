def join_str(joiner="-", content_list=[]):
    """
    Join all items of content_list with the joiner.
    :param joiner: the joiner, "-" by default
    :param content_list: list of items to join; each item must be a str
    :return: content_list joined with the joiner
    """
    return joiner.join(content_list)


def organize_info(key=[], value=[]):
    """
    Reorganize information into the following format:
    [** {key} **]: {value}
    :param key: index names
    :param value: the value corresponding to each index
    :return: the organized string
    """
    assert len(key) == len(value)

    info = []
    length = len(key)
    for idx in range(length):
        info.append("[** " + key[idx] + " **]: " + str(value[idx]))

    return join_str("\n", info)
