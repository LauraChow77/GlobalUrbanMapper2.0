import time

from core.utils.pretty_format import join_str


def get_curr_time():
    """
    获取当前时间
    :return: 当前时间的str
    """
    lt = time.localtime(time.time())
    yyyy = str(lt.tm_year)
    mm = str(lt.tm_mon)
    dd = str(lt.tm_mday)
    hh = str(lt.tm_hour)
    mn = str(lt.tm_min)
    sc = str(lt.tm_sec)

    lt_str = join_str("-", [yyyy, mm, dd, hh, mn, sc])

    return lt_str

