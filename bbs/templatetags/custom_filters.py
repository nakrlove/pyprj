from django import template

register = template.Library()

@register.filter
def repeat(value, count):
    """문자열을 count만큼 반복"""
    return str(value) * int(count)


@register.filter
def desnumber(total,num):
    return total - num;