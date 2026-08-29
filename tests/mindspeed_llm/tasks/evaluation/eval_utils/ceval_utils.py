from typing import List, Tuple

import re
import pandas as pd
from torch import distributed as dist


ceval_subject_mapping = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other'],
}
ceval_all_sets = list(ceval_subject_mapping.keys())


def format_ceval_templates(target_data, support_set, subject_name):
    few_shots_questions = ''
    subject_name = ceval_subject_mapping[subject_name][1]
    target_question_template = "以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: "
    for _, template_question in support_set.iterrows():
        few_shots_questions += parse_templates(template_question, subject_name)
    target_question_template = target_question_template.format(subject_name=subject_name, question=target_data['question'], A=target_data['A'], B=target_data['B'], C=target_data['C'], D=target_data['D'])
    target_question_template = few_shots_questions + target_question_template
    return target_question_template


def parse_templates(row, subject_name):
    template_prefex = "以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: \n{answer}\n"
    question = row['question']
    choice_A = row['A']
    choice_B = row['B']
    choice_C = row['C']
    choice_D = row['D']
    answer = row['answer']
    prompt = template_prefex.format(subject_name=subject_name, question=question, A=choice_A, B=choice_B, C=choice_C, D=choice_D, answer=answer)
    return prompt


def first_capital_postprocess(text: str) -> str:
    if isinstance(text, tuple) or isinstance(text, list):
        text = text[0]
    for t in list(text.strip()):
        if t.isupper():
            return t
    return ''