import PyUber
import pandas as pd
import numpy as np
import dash
from dash import Dash, dcc, html, State, callback
from dash import dash_table as dt
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta

SQL_DATA = '''
SELECT 
          a2.monitor_set_name AS monitor_set_name
         ,a5.value AS chart_value
         ,a5.test_name AS chart_test_name
         ,a0.operation AS spc_operation
         ,a1.entity AS entity
         ,To_Char(a1.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS entity_data_collect_date
         ,a10.centerline AS centerline
         ,a10.lo_control_lmt AS lo_control_lmt
         ,a10.up_control_lmt AS up_control_lmt
         ,CASE WHEN a10.centerline IS NULL THEN -99 WHEN a5.value BETWEEN a10.centerline - ((a10.centerline - a10.lo_control_lmt)/3) AND a10.centerline Then -1 WHEN a5.value BETWEEN a10.centerline AND a10.centerline + ((a10.up_control_lmt - a10.centerline)/3) THEN 1 WHEN a5.value BETWEEN a10.centerline - (2*((a10.centerline - a10.lo_control_lmt)/3)) AND a10.centerline THEN -2 WHEN a5.value BETWEEN a10.centerline AND a10.centerline + (2*((a10.up_control_lmt - a10.centerline)/3)) THEN 2 WHEN a5.value Between a10.lo_control_lmt AND a10.centerline - (2.*((a10.centerline - a10.lo_control_lmt)/3.)) THEN -3 WHEN a5.value Between a10.centerline + (2*((a10.up_control_lmt - a10.centerline)/3)) AND a10.up_control_lmt THEN 3 WHEN a5.value > a10.up_control_lmt THEN 4 WHEN a5.value < a10.lo_control_lmt THEN -4 ELSE 999 END AS zone
         ,a5.spc_chart_category AS spc_chart_category
         ,a5.spc_chart_subset AS spc_chart_subset
         ,a0.lot AS lot
         ,To_Char(a0.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS lot_data_collect_date
         ,a0.route AS route
         ,a3.parameter_class AS parameter_class
         ,a3.measurement_set_name AS measurement_set_name
         ,a2.violation_flag AS violation_flag
         ,a5.valid_flag AS chart_pt_valid_flag
         ,a5.standard_flag AS chart_standard_flag
         ,a5.chart_type AS chart_type
         ,a4.foup_slot AS foup_slot
         ,a4.wafer AS raw_wafer
         ,a4.value AS raw_value
         ,a4.wafer3 AS raw_wafer3
FROM 
P_SPC_ENTITY a1
LEFT JOIN P_SPC_Lot a0 ON a0.spcs_id = a1.spcs_id
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a1.spcs_id AND a2.data_collection_time=a1.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
INNER JOIN P_SPC_CHART_POINT a5 ON a5.spcs_id = a3.spcs_id AND a5.measurement_set_name = a3.measurement_set_name
LEFT JOIN P_SPC_CHARTPOINT_MEASUREMENT a7 ON a7.spcs_id = a3.spcs_id and a7.measurement_set_name = a3.measurement_set_name
AND a5.spcs_id = a7.spcs_id AND a5.chart_id = a7.chart_id AND a5.chart_point_seq = a7.chart_point_seq AND a5.measurement_set_name = a7.measurement_set_name
LEFT JOIN P_SPC_CHART_LIMIT a10 ON a10.chart_id = a5.chart_id AND a10.limit_id = a5.limit_id
LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
AND a4.spcs_id = a7.spcs_id AND a4.measurement_id = a7.measurement_id
WHERE
 (a2.monitor_set_name Like '%DSA_PST_NONPAT.5051.MON' or a2.monitor_set_name Like '%DSA_PST.5051.MON')
 AND      a0.operation In ('8281','8333') 
 AND      a1.entity Like 'T%' 
 AND      a1.data_collection_time >= SYSDATE - 30
'''

SQL_SLOTS = '''
SELECT 
          leh.lot AS lot1,
          wch.wafer AS wafer,
          wch.waf3 AS waf3,
          leh.entity AS lot_entity,
          wch.slot AS slot,
          wch.chamber AS chamber,
          wch.state AS state,
          To_Char(wch.start_time,'yyyy-mm-dd hh24:mi:ss') AS start_date,
          To_Char(wch.end_time,'yyyy-mm-dd hh24:mi:ss') AS end_date,
          leh.operation AS operation1,
          lwr2.recipe AS lot_recipe,
          To_Char(leh.introduce_txn_time,'yyyy-mm-dd hh24:mi:ss') AS lot_introduce_txn_date
FROM 
          F_LotEntityHist leh
INNER JOIN
          F_WaferChamberHist wch
ON 
          leh.runkey = wch.runkey
INNER JOIN 
          F_Lot_Wafer_Recipe lwr2 
ON 
          lwr2.recipe_id = leh.lot_recipe_id
WHERE
          (wch.chamber LIKE '%ADH%' OR
           wch.chamber LIKE '%HP%' OR
           wch.chamber LIKE '%PH%' OR
           wch.chamber LIKE '%CPHG%' OR
           wch.chamber LIKE '%RGCH%' OR
           wch.chamber LIKE '%CGCH%' OR
           wch.chamber LIKE '%CPL%' OR
           wch.chamber LIKE '%WEE%' OR
           wch.chamber LIKE '%ITC%' OR
           wch.chamber LIKE '%BCT%' OR
           wch.chamber LIKE '%COT%' OR
           wch.chamber LIKE '%PCT%' OR
           wch.chamber LIKE '%DEV%') 
AND 
          (leh.entity LIKE 'SDJ591' OR
           leh.entity LIKE 'SCJ591' OR
           leh.entity LIKE 'SBH202' OR
           leh.entity LIKE 'SDJ111' OR
           leh.entity LIKE 'TZH591' OR
           leh.entity LIKE 'TBC611')
AND 
          leh.run_txn_time >= TRUNC(SYSDATE) - 31
AND 
          (leh.lot LIKE 'LN1718%' OR
           leh.lot LIKE 'BARC%')
AND 
          (lwr2.recipe LIKE 'C-%' OR
           lwr2.recipe LIKE 'ZC-%' OR
           lwr2.recipe LIKE 'Spire/C-%' OR
           lwr2.recipe LIKE 'Spire/ZC-%' OR
           lwr2.recipe LIKE 'RADIAL%')
'''

try:
    conn = PyUber.connect(datasource='F21_PROD_XEUS')
    df = pd.read_sql(SQL_DATA, conn)
    df_chamber = pd.read_sql(SQL_SLOTS, conn)
except:
    print('Cannot run SQL script - Consider connecting to VPN')

df_chamber = df_chamber.rename(columns={'LOT1': 'LOT'})
df_chamber = df_chamber.rename(columns={'LOT_RECIPE': 'RCP'})
df_chamber = df_chamber.rename(columns={'LOT_INTRODUCE_TXN_DATE': 'INTRO_DATE'})
df_chamber = df_chamber.rename(columns={'LOT_ENTITY': 'ENTITY'})
df_chamber = df_chamber.rename(columns={'OPERATION1': 'OPN'})
df_chamber['OPN'] = df_chamber['OPN'].astype(str)
df_chamber = df_chamber.drop(columns=['WAFER', 'START_DATE', 'END_DATE'], errors='ignore')
# Sort df_chamber by DATE in descending order
df_chamber = df_chamber.sort_values(by='INTRO_DATE', ascending=True)
# Remove '-V' or '-v' suffix and trailing spaces from the 'RCP' column
df_chamber['RCP'] = df_chamber['RCP'].str.replace(r'(-V|-v)\s*$', '', regex=True).str.strip()


#create columns for chamber data to go from tall to wide format
bake_num = 3
chill_num = 4
chamberl = ['ADH', 'COT', 'ITC', 'BCT', 'PCT', 'DEV']
ch_columns = [f'{stage}' for stage in chamberl + [f'BAKE{i}' for i in range(1, bake_num + 1)] + [f'CHILL{i}' for i in range(1, chill_num + 1)]]
for col in ch_columns:
    df_chamber[col] = None

# Flatten the CHAMBER column
def flatten_chamber(group):
    adh = cot = itc = bct = pct = dvlp = None
    bake = [None] * bake_num
    chill = [None] * chill_num
    bake_count = 0
    chill_count = 0
    for idx, row in group.iterrows():
        ch = row['CHAMBER']
        if ch.startswith(('ADH', 'CADH')):
            adh = ch
        elif ch.startswith('COT'):
            cot = ch
        elif ch.startswith('ITC'):
            itc = ch
        elif ch.startswith('BCT'):
            bct = ch
        elif ch.startswith('PCT'):
            pct = ch
        elif ch.startswith('DEV'):
            dvlp = ch
        elif ch.startswith(('RGCH', 'CGCH', 'CPHG', 'CPHP', 'PHP')):
            if bake_count < bake_num:
                bake[bake_count] = ch
                bake_count += 1
        elif ch.startswith(('CPL', 'SCPL', 'CLHP')):
            if chill_count < chill_num:
                chill[chill_count] = ch
                chill_count += 1
    group['ADH'] = adh
    group['COT'] = cot
    group['ITC'] = itc
    group['BCT'] = bct
    group['PCT'] = pct
    group['DEV'] = dvlp
    for i in range(bake_num):
        group[f'BAKE{i+1}'] = bake[i]
    for i in range(chill_num):
        group[f'CHILL{i+1}'] = chill[i]
    return group.iloc[0]

# call the chamber function. important to groupby INTRO_DATE since the same lot, wfr can be used multiple times
df_chamber = df_chamber.groupby(['ENTITY', 'INTRO_DATE', 'OPN', 'LOT', 'SLOT', 'WAF3', 'RCP'], group_keys=False).apply(flatten_chamber).reset_index(drop=True)
df_chamber = df_chamber.drop(columns=['CHAMBER'])

df_chamber = df_chamber.sort_values(by=['INTRO_DATE', 'SLOT'], ascending=[False, False])

df_chamber.to_excel('df_chamber.xlsx', index=False)

# Define the configurations for each chart. These are the columns to be removed from the DataFrame. For example DEV is not used on the coat only monitors.
configSTD = ['ITC', 'BCT', 'PCT', 'DEV']
configBARC = ['ADH', 'COT', 'ITC', 'PCT', 'DEV','BAKE2', 'BAKE3', 'CHILL3', 'CHILL4']
config248 = ['PCT', 'DEV']
config193 = ['ITC', 'BCT', 'PCT', 'DEV']
configILINE = ['ITC', 'BCT', 'PCT', 'DEV']
#all = ['ADH', 'COT', 'ITC', 'BCT', 'PCT', 'DEV', 'BAKE1', 'BAKE2', 'BAKE3', 'CHILL1', 'CHILL2', 'CHILL3', 'CHILL4']

chart_config_lookup = {
    '5051B9110D1600DFX': [configBARC, ['Spire/C-9110D-1600-0A', 'C-9110D-1600-0A']],
    '5051B9145430DFX': [configBARC, ['Spire/C-9145-430-0A', 'C-9145-430-0A']],
    '5051B9825280DFX': [configBARC, ['Spire/C-9825-280-0A', 'C-9825-280-0A']],
    '5051B29A780DFX': [configBARC, ['Spire/C-ARC29-780-0A', 'C-ARC29-780-0A']],
    '5051B42P1100DFX': [configBARC, ['Spire/C-DUV42-1100-0A', 'C-DUV42-1100-0A', 'C-DUV42-1100-1A', 'C-DUV42-1100-2A']],
    '50512419DFX': [configSTD, ['Spire/C-2419-2300-10A']],
    '50513522DFX': [configSTD, ['Spire/C-3522-2000-05B']],
    '50517773CDFX': [configSTD, ['Spire/C-7773-1505-05A']],
    '5051A12SDFX': [configSTD, ['Spire/C-A12S-1390-05A']],
    '5051460235500DFX': [config248, ['Spire/C-4602-35500-10A']],
    '5051460413600DFX': [config248, ['Spire/C-4604-13600-10A']],
    '50517152C197600DFX': [config248, ['Spire/C-7152C19-7600-A']],
    '5051M4844200DFX': [config248, ['Spire/C-M484-4200-10A']],
    '5051P8023850DFX': [config248, ['Spire/C-P802-3850-05A']],
    '5051TCX420DFX': [config248, ['Spire/C-TCX-420-A']],
    '5051IX215DFX': [configSTD, ['Spire/C-215-22K-10A']],
    '5051IX428DFX': [configSTD, ['Spire/C-428-10K-25A']],
    '5051IN009DFX': [configSTD, ['Spire/C-iN009-34K-27A']]
}

# Function to get RCP values for a given CHART_TEST_NAME
def get_rcp_values(chart_test_name):
    config_and_rcp_list = chart_config_lookup.get(chart_test_name, None)
    if config_and_rcp_list:
        config, rcp_list = config_and_rcp_list
        return rcp_list
    else:
        return []

# Create a dictionary to store the DataFrames
df_dict = {}

# Iterate over the keys in chart_config_lookup
for chart_test_name in chart_config_lookup.keys():
    rcp_values = get_rcp_values(chart_test_name)
    # Filter df_chamber based on RCP values
    filtered_df = df_chamber[df_chamber['RCP'].isin(rcp_values)]
    # Store the filtered DataFrame in the dictionary
    df_dict[chart_test_name] = filtered_df

# Remove columns from each DataFrame in df_dict based on the config in chart_config_lookup
for chart_test_name, (config, rcp_list) in chart_config_lookup.items():
    if chart_test_name in df_dict:
        df_dict[chart_test_name] = df_dict[chart_test_name].drop(columns=config, errors='ignore')




# Extract the RESIST value and create a new column
df['RESIST'] = df['SPC_CHART_CATEGORY'].str.extract(r'RESIST=([^;]+)')
# Remove the 'PARTICLE_SIZE=' portion from the 'SPC_CHART_SUBSET' column values
df['SPC_CHART_SUBSET'] = df['SPC_CHART_SUBSET'].str.replace('PARTICLE_SIZE=', '')

# Rename columns
df.rename(columns={'VIOLATION_FLAG': 'FAIL', 'CHART_PT_VALID_FLAG': 'VALID_FLAG', 'CHART_STANDARD_FLAG': 'STD_FLAG'}, inplace=True)
# Create the VALID column based on VALID_FLAG and STD_FLAG
df['VALID'] = df.apply(lambda row: 'N' if row['VALID_FLAG'] == 'N' or row['STD_FLAG'] == 'N' else 'Y', axis=1)

# Adjust ENTITY_DATA_COLLECT_DATE for each group of unique ENTITY, ENTITY_DATA_COLLECT_DATE, RAW_WAFER
df['ENTITY_DATA_COLLECT_DATE'] = pd.to_datetime(df['ENTITY_DATA_COLLECT_DATE'])  # Ensure the column is in datetime format
df = df.sort_values(by=['ENTITY', 'ENTITY_DATA_COLLECT_DATE', 'FOUP_SLOT'])  # Sort the DataFrame for consistent ordering

# Hover was not showing the different wafers for a specific run so Increment ENTITY_DATA_COLLECT_DATE by 1 minute for each wfr so you can hover over each separately
df['ENTITY_DATA_COLLECT_DATE'] += df.groupby(['ENTITY', 'ENTITY_DATA_COLLECT_DATE', 'SPC_CHART_SUBSET']).cumcount().apply(
    lambda x: timedelta(minutes=x)
)

df.to_excel('df_data.xlsx', index=False)

#df_data.to_csv('df_data.csv', index=False)

# Load data for testing
#df = pd.read_csv('df_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Get unique defect_size
defect_sizes = df['SPC_CHART_SUBSET'].unique()

# Layout with radio button for filtering valid data
app.layout = html.Div([
    dcc.RadioItems(
        id='only-valid',
        options=[
            {'label': 'Only Valid Data', 'value': 'Y'},
            {'label': 'All Data', 'value': 'N'}
        ],
        value='Y',  # Default value
        labelStyle={'display': 'inline-block'}
    ),
    dcc.RadioItems(
        id='y-axis-scale',  # New radio button for y-axis scaling
        options=[
            {'label': 'Auto Scale', 'value': 'auto'},
            {'label': 'Use Upper Limit', 'value': 'upper_limit'}
        ],
        value='auto',  # Default value
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(id='charts-container')  # Container for the charts
])

# Callback to update charts based on radio button selection
@app.callback(
    Output('charts-container', 'children'),
    [Input('only-valid', 'value'),
     Input('y-axis-scale', 'value')]  # New input for y-axis scaling
)
def update_charts(only_valid, y_axis_scale):
    rows = []
    added_tables = set()  # Set to keep track of added tables
    for resist in df['CHART_TEST_NAME'].unique():
        resist_df = df[df['CHART_TEST_NAME'] == resist]
        if only_valid == 'Y':
            resist_df = resist_df[resist_df['VALID'] == 'Y']
        columns = []
        for defect_size in defect_sizes:
            defect_size_df = resist_df[resist_df['SPC_CHART_SUBSET'] == defect_size]
            if not defect_size_df.empty:
                last_monitor_set_name = defect_size_df['MONITOR_SET_NAME'].iloc[-1]
                last_chart_test_name = defect_size_df['CHART_TEST_NAME'].iloc[-1]
                last_measurement_set_name = defect_size_df['MEASUREMENT_SET_NAME'].iloc[-1]
                resist_name = defect_size_df['RESIST'].iloc[-1]
                title = f'{resist_name} - {defect_size}<br>{last_monitor_set_name} {last_chart_test_name}<br>{last_measurement_set_name}'
                upper_limit = 2 * defect_size_df['UP_CONTROL_LMT'].iloc[-1]
                center_line = defect_size_df['CENTERLINE'].iloc[-1]
                fig = px.line(defect_size_df, x='ENTITY_DATA_COLLECT_DATE', y='RAW_VALUE', title=title, color='ENTITY')
                fig.add_hline(y=defect_size_df['UP_CONTROL_LMT'].iloc[-1], line_dash="dash", annotation_text="Upper Spec", line_color="red")
                if pd.notna(center_line) and center_line != '':
                    fig.add_hline(y=center_line, line_dash="dash", annotation_text="Center Line")
                hovertext = [f'Date: {date}<br>LOT: {lot}<br>ROUTE: {route}<br>VALUE: {raw_value}<br>FAIL: {fail}<br>VALID: {valid}<br>FOUP_SLOT: {foup_slot}<br>RAW_WAFER3: {raw_wafer3}' 
                            for date, lot, route, raw_value, fail, valid, foup_slot, raw_wafer3 in zip(
                                defect_size_df['ENTITY_DATA_COLLECT_DATE'], 
                                defect_size_df['LOT'], 
                                defect_size_df['ROUTE'], 
                                defect_size_df['RAW_VALUE'], 
                                defect_size_df['FAIL'], 
                                defect_size_df['VALID'], 
                                defect_size_df['FOUP_SLOT'], 
                                defect_size_df['RAW_WAFER3']
                            )]
                valid_symbols = defect_size_df['VALID'].map({'Y': 'circle', 'N': 'x'})
                fig.update_traces(mode='markers', hovertemplate='%{hovertext}', hovertext=hovertext, marker=dict(symbol=valid_symbols))
                 # Update the layout with the y-axis range based on the selected scaling mode
                if y_axis_scale == 'upper_limit':
                    fig.update_layout(
                        yaxis_range=[0, upper_limit],  # Use explicit upper limit
                    )
                else:
                    fig.update_layout(
                        yaxis_autorange=True  # Enable auto-scaling
                    )
                columns.append(html.Div(dcc.Graph(figure=fig), className='column', style={'flex': '1'}))
        
        # Add table for the current CHART_TEST_NAME if not already added
        if resist not in added_tables:
            table_df = df_dict.get(resist)
            if table_df is not None:
                table = dt.DataTable(
                    columns=[{"name": i, "id": i} for i in table_df.columns],
                    data=table_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    filter_action="native"  # Make columns filterable
                )
                columns.append(html.Div(table, className='column', style={'flex': '1'}))
                added_tables.add(resist)  # Mark table as added
        
        rows.append(html.Div(columns, className='row', style={'display': 'flex', 'flexDirection': 'row'}))

    return html.Div(rows)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)