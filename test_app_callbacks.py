from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
from app import calculate_value, update_total_count

def test_calculate_value_callback_one():
    output = calculate_value("Conference")
    assert output == 545

def test_calculate_value_callback_two():
    output = calculate_value("Journal")
    assert output == 77

def test_update_total_count_one():
    output = update_total_count("Conference")
    assert output == 'Total Count: 545'

def test_update_total_count_two():
    output = update_total_count("Journal")
    assert output == 'Total Count: 77'