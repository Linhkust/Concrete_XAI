from shiny import App, ui, reactive, render
import pandas as pd
import faicons as fa
from train import Ml_model
import joblib
import os
from visualization import taylor_diagram
from XAI import Importance
from llm_analysis import llm_explain

ICONS = {
    "pipeline": fa.icon_svg("timeline"),
    "time": fa.icon_svg("clock"),
    'target': fa.icon_svg('ranking-star'),
    'play': fa.icon_svg('play'),
    'bots': fa.icon_svg('bots')
}

app_ui = ui.page_navbar(
    # Model Generation and Evaluation
    ui.nav_panel("Model Generation & Evaluation",
                 ui.page_sidebar(
                     ui.sidebar(
                         # data Upload
                         ui.card(
                             ui.card_header('Step 1: Data Upload'),
                             ui.input_file("final", "Upload your data", accept='.csv', placeholder='No file selected'),
                             ui.input_select('target_col', 'Target column', choices=[]),
                         ),

                         # train test split
                         ui.card(
                             ui.card_header('Step 2: Model Training Configuration'),
                             ui.input_slider("train_test", "Training data size", 0, 1, 0.7, step=0.05),
                         ),

                         # configuration of AutoML frameworks
                         ui.input_task_button('model_training', 'Model Train'),
                         ui.download_button('download1', 'Download trained model', class_="btn-success"),
                         width=500, open='always'
                     ),

                     # Training results
                     ui.layout_columns(
                         ui.value_box("Total Training Time", ui.output_ui("training_time"),
                                      showcase=ICONS['time']),
                         ui.value_box("Best Model", ui.output_ui("best_pipeline"),
                                      showcase=ICONS['target']),
                         fill=False,
                     ),

                     # model performance leaderboard
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Model Performance Leaderboard'),
                             ui.output_data_frame('leaderboard'),
                             height='600px'
                         )
                     ),

                     # taylor diagram
                     ui.input_task_button('draw_taylor', 'Draw Taylor diagram'),
                     ui.output_ui('Taylor_diagram'),
                 ),
                 ),

    # XAI
    ui.nav_panel("Explainable Artificial Intelligence",
                 ui.page_sidebar(
                     ui.sidebar(

                         # Data upload
                         ui.card(
                             ui.card_header('Step 1: Data Upload'),
                             ui.input_file("xai_data", "Upload the data", accept=['.csv'],
                                           placeholder='No file selected'),
                             ui.input_select('xai_target_col', 'Target column', choices=[]),
                         ),
                            ui.card(
                                            ui.card_header('Step 2: Model Upload'),
                                            ui.input_file("pkl_upload", "Upload ML model", accept=['.pkl'],
                                                          placeholder='No file selected'),
                                        ),
                         # # Model upload
                         # ui.input_switch("pkl", "Use your own model", False),
                         # ui.output_ui("pkl_upload"),
                         # Model explain
                         ui.input_task_button('model_explain', 'Model Explain'),
                         width=500, open='always'
                     ),

                     ui.navset_card_tab(
                         ui.nav_panel("PFI plot",
                                       ui.output_plot('pfi')
                                    ),

                         ui.nav_panel("SHAP Summary Plot",
                                      ui.output_plot('shap_global')
                                      ),

                         ui.nav_panel("PDP plot & SHAP Dependence Plot",
                                      ui.input_select('xai_feature', 'Selected ONE Feature',
                                     choices=[],
                                     multiple=False,
                                     ),
                                      ui.input_task_button("xai_plot", "1D Plot", width='300px'),
                                      ui.layout_columns(
                                          # PDP plot
                                          ui.card(
                                              ui.card_header('1D Partial Dependence Plot'),
                                              ui.output_plot('pdp1'),
                                              height='500px'),

                                          # 1D SHAP depedence plot
                                          ui.card(
                                              ui.card_header('1D SHAP Dependence Plot'),
                                              ui.output_plot('shap_dp1'),
                                              height='500px'),
                                      ),
                                      # 2D plot
                                     ui.input_selectize('xai_features', 'Selected TWO Features',
                                     choices=[],
                                     multiple=True,
                                     ),
                                     ui.input_task_button("xai_plot2", "2D Plot", width='300px'),
                                     ui.layout_columns(
                                          # 2D PDP plot
                                          ui.card(
                                              ui.card_header('2D Partial Dependence Plot'),
                                              ui.output_plot('pdp2'),
                                              height='500px'),

                                          # 2D SHAP depedence plot
                                          ui.card(
                                              ui.card_header('2D SHAP Dependence Plot'),
                                              ui.output_plot('shap_dp2'),
                                              height='500px'),
                                      ),
                                      ),
                         id="tab",
                     ),
                 ),
                 ),

    # LLM analysis
    ui.nav_panel("Large Language Model Analysis",
                 ui.page_sidebar(
                     ui.sidebar(
                         ui.card(
                             ui.card_header('Step 1: XAI plot upload'),
                             # image preview
                             ui.input_file("xai_image", "Upload the XAI plot", accept=['.png', '.jpg'],
                                           placeholder='No file selected'),
                             # image type
                             ui.input_select(
                                 "plot_type",
                                 "Select corresponding XAI plot type:",
                                 ['permutation feature importance plot',
                                  'SHAP summary plot',
                                  '1D Partial dependence plot',
                                  '1D SHAP dependence plot',
                                  '2D Partial dependence plot',
                                  '2D SHAP dependence plot'])
                                      ),
                            # LLM configuration
                            ui.card(ui.card_header('Step 2: LLM Configuration'),
                            ui.input_text("llm_model", "LLM Model", "e.g., qwen-vl-plus"),
                            ui.input_text("llm_base_url", "Base URL", "e.g., https://dashscope.aliyuncs.com/compatible-mode/v1"),
                            ui.input_text("llm_api_key", "API Key", "Enter your api key")
                                    ),
                         ui.input_task_button('llm_analysis', 'LLM Analysis'),
                         width=500, open='always'
                     ),

                     # image show and llm analysis results
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Image preview'),
                             ui.output_image('xai_view')
                           )
                            ),
                     ui.output_text_verbatim('llm_results')
                 )
                 ),
    title="[Concrete_XAI]",
    id="page",
)


def server(input, output, session):

    '''
    Model Pipeline Generation & Evaluation
    '''
    # update target column
    @reactive.effect
    def update_target_col():
        if input.final() is not None:
            ui.update_select(
                'target_col',
                choices=pd.read_csv(input.final()[0]["datapath"]).columns.tolist())

    # Draw the Taylor diagram
    @render.ui
    @reactive.event(input.draw_taylor)
    def Taylor_diagram():
        return ui.layout_columns(
                     ui.card('Taylor Diagram (trian set)',
                         ui.output_plot('taylor_plot_train'), height='900px'
                     ),
                     ui.card('Taylor Diagram (test set)',
                         ui.output_plot('taylor_plot_test'), height='900px'
                     ),
                 )

    @render.data_frame
    @reactive.event(input.model_training, ignore_none=False)
    def leaderboard():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            data = pd.read_csv(path)
            # download_path = os.path.dirname(path) + '_feature.csv'
            p_board, model = Ml_model(df=data, target=input.target_col(), train_size=input.train_test())._performance()
            p_board.to_csv(os.path.dirname(path) + '\\results.csv', index=False)
            joblib.dump(model, os.path.dirname(path) + '\\best_model.pkl')
            return  pd.read_csv(os.path.dirname(path) + '\\results.csv').drop(columns=['Training time', 'Best params'])

    @render.ui
    @reactive.event(input.model_training, ignore_none=False)
    def training_time():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            results = pd.read_csv(os.path.dirname(path) + '\\results.csv')
            time = '%.1f' % (results['Training time'].sum())
            return '{} min'.format(time)

    @render.ui
    @reactive.event(input.model_training, ignore_none=False)
    def best_pipeline():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            results = pd.read_csv(os.path.dirname(path) + '\\results.csv')

            train_df = results[results['Data type'] == 'Train set']
            test_df =results[results['Data type'] == 'Test set']
            max_row = test_df.loc[test_df['R2'].idxmax()]
            best_method = max_row['Method']
            best_params = train_df[train_df['Method'] == best_method].reset_index(drop=True).loc[0, 'Best params']
            # print(best_method)
            # print(best_params)
            if best_method == 'LR':
                return best_method
            else:

                return best_method + ' with best hyperparameters: ' + best_params

    @render.plot
    @reactive.event(input.draw_taylor, ignore_none=False)
    def taylor_plot_train():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            results = pd.read_csv(os.path.dirname(path) + '\\results.csv')
            train_df = results[results['Data type'] == 'Train set'].reset_index(drop=True)
            fig = taylor_diagram(train_df)
            return fig

    @render.plot
    @reactive.event(input.draw_taylor, ignore_none=False)
    def taylor_plot_test():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            results = pd.read_csv(os.path.dirname(path) + '\\results.csv')
            test_df = results[results['Data type'] == 'Test set'].reset_index(drop=True)
            fig = taylor_diagram(test_df)
            return fig

    @render.download()
    def download1():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            return os.path.dirname(path) + '\\best_model.pkl'

    '''Explainable Artificial Intelligence'''

    # update target column
    @reactive.effect
    def update_xai_col():
        if input.xai_data() is not None:
            ui.update_select(
                'xai_target_col',
                choices=pd.read_csv(input.xai_data()[0]["datapath"]).columns.tolist())

    # @render.ui
    # @reactive.event(input.pkl)
    # def pkl_upload():
    #     if input.pkl():
    #         return ui.card(
    #             ui.card_header('Model Upload'),
    #             ui.input_file("pkl_upload", "Upload your model", accept=['.pkl'],
    #                           placeholder='No file selected'),
    #         )

    def xai_model():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            fi = Importance(df=data, target=input.xai_target_col())
            return fi

    # pfi feature importance
    @render.plot
    @reactive.event(input.model_explain)
    def pfi():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().pfi(model=joblib.load(model), df_type='all')
            return  ax

    # shap summary plot
    @render.plot
    @reactive.event(input.model_explain)
    def shap_global():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().shap_summary(model=joblib.load(model), df_type='all')
            return ax

    # update selected features
    @reactive.effect
    def _():
        if input.xai_data() is not None:
            choices = pd.read_csv(input.xai_data()[0]["datapath"]).columns.tolist()
            ui.update_selectize(
                'xai_feature',
                choices=choices, )
            ui.update_selectize(
                'xai_features',
                choices=choices, )

    # pdp
    @render.plot
    @reactive.event(input.xai_plot)
    def pdp1():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().pdp(model=joblib.load(model), features=[input.xai_feature()], df_type='all')
            return ax

    # shap_dp
    @render.plot
    @reactive.event(input.xai_plot)
    def shap_dp1():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().shap_scatter_1d(joblib.load(model), variable=input.xai_feature(), df_type='all')
            return ax

    # pdp
    @render.plot
    @reactive.event(input.xai_plot2)
    def pdp2():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().pdp(joblib.load(model), features=[input.xai_features()], df_type='all')
            return ax

    # shap_dp
    @render.plot
    @reactive.event(input.xai_plot2)
    def shap_dp2():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            ax = xai_model().shap_scatter_2d(joblib.load(model),
                                               variable=input.xai_features()[0],
                                               interact_term=input.xai_features()[1], df_type='all')
            return ax

    '''LLM analysis'''
    @render.image
    @reactive.event(input.llm_analysis)
    def xai_view():
        if input.xai_image()[0]['datapath'] is not None:
            img = {"src": input.xai_image()[0]['datapath']}
            return img

    @render.text
    @reactive.event(input.llm_analysis)
    def llm_results():
        path = input.xai_image()[0]['datapath']
        return llm_explain(image_path=path,
                           image_type=input.plot_type(),
                           api_key=input.llm_api_key(),
                           base_url=input.llm_base_url(),
                           model=input.llm_model())


app = App(app_ui, server)
app.run()