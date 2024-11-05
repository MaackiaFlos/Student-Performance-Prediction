import pandas as pd

df = {}
csv_file_paths = {
    r'E:\Student_Dataset\anonymiseddata\vle.csv': 'vle',
    r'E:\Student_Dataset\anonymiseddata\studentVle.csv': 'studentVle',
    r'E:\Student_Dataset\anonymiseddata\studentRegistration.csv': 'studentRegistration',
    r'E:\Student_Dataset\anonymiseddata\studentInfo.csv': 'studentInfo',
    r'E:\Student_Dataset\anonymiseddata\studentAssessment.csv': 'studentAssessment',
    r'E:\Student_Dataset\anonymiseddata\courses.csv': 'courses',
    r'E:\Student_Dataset\anonymiseddata\assessments.csv': 'assessments',
}

for file, table_name in csv_file_paths.items():
    df[table_name] = pd.read_csv(file)

# Filtering and preparing the data
exams = df['assessments'][df['assessments']["assessment_type"] == "Exam"]
others = df['assessments'][df['assessments']["assessment_type"] != "Exam"]
amounts = others.groupby(["code_module", "code_presentation"]).count()["id_assessment"].reset_index()

# Function to classify pass/fail based on grade
def pass_fail(grade):
    return True if grade >= 40 else False

# Creating dataframe for assessments with weights and grades
stud_ass = pd.merge(df['studentAssessment'], others, how="inner", on=["id_assessment"])
stud_ass["pass"] = stud_ass["score"].apply(pass_fail)
stud_ass["weighted_grade"] = stud_ass["score"] * stud_ass["weight"] / 100

# Final assessment average per student per module
avg_grade = stud_ass.groupby(["id_student", "code_module", "code_presentation"]).sum()["weighted_grade"].reset_index()

# Pass rate per student per module
pass_rate = pd.merge(
    (stud_ass[stud_ass["pass"] == True].groupby(["id_student", "code_module", "code_presentation"]).count()["pass"]).reset_index(),
    amounts, how="left", on=["code_module", "code_presentation"]
)
pass_rate["pass_rate"] = pass_rate["pass"] / pass_rate["id_assessment"]
pass_rate.drop(["pass", "id_assessment"], axis=1, inplace=True)

# Final exam scores
stud_exams = pd.merge(df['studentAssessment'], exams, how="inner", on=["id_assessment"])
stud_exams["exam_score"] = stud_exams["score"]
stud_exams.drop(["id_assessment", "date_submitted", "is_banked", "score", "assessment_type", "date", "weight"], axis=1, inplace=True)

# Filtering studentInfo and keeping relevant columns
df['studentInfo'] = df['studentInfo'][df['studentInfo']["final_result"] != "Withdrawn"]
df['studentInfo'] = df['studentInfo'][["code_module", "code_presentation", "id_student", "num_of_prev_attempts", "final_result"]]

# Merging all dataframes into final_df
df_1 = pd.merge(avg_grade, pass_rate, how="inner", on=["id_student", "code_module", "code_presentation"])
assessment_info = pd.merge(df_1, stud_exams, how="inner", on=["id_student", "code_module", "code_presentation"])
df_2 = pd.merge(df['studentInfo'], assessment_info, how="inner", on=["id_student", "code_module", "code_presentation"])
final_df = pd.merge(df_2, df['studentVle'].groupby(["id_student", "code_module", "code_presentation"]).mean()[["date", "sum_click"]].reset_index(), how="inner", on=["id_student", "code_module", "code_presentation"])

# Dropping unnecessary columns
final_df.drop(["id_student", "code_module", "code_presentation"], axis=1, inplace=True)

final_df.to_csv("final_df.csv", index = False)