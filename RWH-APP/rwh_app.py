import flet as ft
import pandas as pd
import numpy as np
import joblib

def main(page: ft.Page):
    page.title = "老旧小区雨水改造经济性评价"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 400
    page.window_height = 800
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20

    try:
        saved_data = joblib.load('rf_bcr_model.joblib')
        model = saved_data['model']
        feature_names = saved_data['features'] # 获取12个中文列名
        df = pd.read_excel('clean_data.xlsx')
        X_hist = df.drop('BCR', axis=1)
        y_hist = df['BCR']
        
        data_min = X_hist.min().values
        data_max = X_hist.max().values
        data_q01 = X_hist.quantile(0.01).values
        data_q99 = X_hist.quantile(0.99).values
    except Exception as e:
        page.add(ft.Text(f"加载失败，请确保模型和Excel在同一目录: {e}", color="red"))
        return

    default_values = [0.45, 1700, 1.2, 800, 6000, 12, 120, 18, 40, 30, 0.10, 4.5]
    input_fields = []

    page.add(
        ft.Text("老旧小区雨水改造经济性评价", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
        ft.Text("基于随机森林模型预测", size=14, color=ft.Colors.GREY_700),
        ft.Divider(height=10, color="transparent"),
    )

    for i in range(12):
        tf = ft.TextField(
            label=feature_names[i], 
            value=str(default_values[i]), 
            keyboard_type=ft.KeyboardType.NUMBER,
            dense=True
        )
        input_fields.append(tf)
        page.add(tf)

    result_card = ft.Container(
        content=ft.Column([
            ft.Text("预测结果", size=18, weight=ft.FontWeight.BOLD),
            ft.Text("BCR: --", size=30, weight=ft.FontWeight.BOLD),
            ft.Text("判定: --", size=20),
            ft.Text("", size=12, color=ft.Colors.ORANGE_800),
            ft.Divider(),
            ft.Text("相似历史项目参考:", size=12, weight=ft.FontWeight.BOLD),
            ft.Text("暂无", size=12)
        ]),
        padding=15,
        bgcolor=ft.Colors.BLUE_50,
        border_radius=10,
        visible=False
    )
    page.add(result_card)
        
    def predict_click(e):
        try:
            inputs = [float(tf.value) for tf in input_fields]
            current_X = np.array([inputs])
            
            pred_bcr = float(model.predict(current_X)[0])
            is_feasible = pred_bcr > 1.0
            
            distances = np.linalg.norm(X_hist.values - current_X, axis=1)
            top3_idx = np.argsort(distances)[:3]
            
            similar_str = ""
            for idx in top3_idx:
                similar_str += f"• {feature_names[0]}:{X_hist.iloc[idx, 0]:.2f}, 真实BCR:{y_hist.iloc[idx]:.2f}\n"
            
            warnings = []
            for i, val in enumerate(inputs):
                if val < data_q01[i] or val > data_q99[i]:
                    warnings.append(feature_names[i])
            
            bcr_text = result_card.content.controls[1]
            judge_text = result_card.content.controls[2]
            warning_text = result_card.content.controls[3]
            similar_text = result_card.content.controls[6]
            
            bcr_text.value = f"BCR: {pred_bcr:.3f}"
            bcr_text.color = ft.Colors.GREEN_700 if is_feasible else ft.Colors.RED_700
            
            judge_text.value = "✅ 经济可行" if is_feasible else "❌ 经济不可行"
            judge_text.color = ft.Colors.GREEN_700 if is_feasible else ft.Colors.RED_700
            
            if warnings:
                warning_text.value = f"⚠️ 外推风险：{', '.join(warnings)} 超出历史 1%-99% 分位"
            else:
                warning_text.value = ""
                
            similar_text.value = similar_str
            result_card.visible = True
            page.update()
            
        except ValueError:
            page.show_snack_bar(ft.SnackBar(ft.Text("请输入有效的数字！")))

    page.add(
        ft.Container(
            content=ft.ElevatedButton(
                "开始预测评估", 
                on_click=predict_click, 
                width=300, 
                height=50,
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
            ),
            alignment=ft.alignment.center,
            margin=ft.margin.only(top=20, bottom=20)
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
               
