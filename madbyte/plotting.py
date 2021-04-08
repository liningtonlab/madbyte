from math import sqrt

import networkx as nx
from bokeh.io import output_file, save
from bokeh.models import (BoxZoomTool, Circle, Column, CustomJS, HoverTool,
                          MultiLine, Plot, Range1d, RangeSlider, ResetTool,
                          TapTool, WheelZoomTool)
from bokeh.plotting.graph import from_networkx


def create_bokeh(G, title, fname, extract_size=15, feature_size=10):
    # Add attributes this way
    try:
        max_weight = max(nx.get_edge_attributes(G, 'weight').values())
    except ValueError:
        max_weight = 1.0
    calc_alpha = lambda x: 0.1 + 0.6 * (x / max_weight)
    edge_attrs = {(s,e): calc_alpha(d['weight']) for s,e,d in G.edges(data=True)}

    node_attrs = {
        k: {
            "_size": feature_size if v['_type']=='spin' else extract_size,
            "_num_members": len(eval(v.get('members', '[]')))
        }
        for k,v in G.nodes(data=True) }
    nx.set_edge_attributes(G, edge_attrs, "_alpha")
    nx.set_node_attributes(G, node_attrs)

    # Show with Bokeh
    plot = Plot(plot_width=1100, plot_height=700,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.title.text = title

    node_hover_tool = HoverTool(tooltips=[("Name", "@index"), ("Members", "@members")])
    plot.add_tools(node_hover_tool, WheelZoomTool(), BoxZoomTool(), ResetTool(), TapTool())

    # Tried this but default layout is just better...
    # def layout(x, **kwargs):
    #     return nx.spring_layout(x, **kwargs, k=1.1/sqrt(x.number_of_nodes()),iterations=10000)
    # graph_renderer = from_networkx(G, layout, scale=1, center=(0, 0))

    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size='_size', fill_color="_color")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="orange")#works
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="red")#works
    graph_renderer.edge_renderer.glyph = MultiLine(line_alpha="_alpha", line_width=1)
    plot.renderers.append(graph_renderer)

    # Add filter slider
    df = nx.to_pandas_edgelist(G)
    max_members = max(nx.get_node_attributes(G,"_num_members").values())
    backup_node_data = graph_renderer.node_renderer.data_source.data
    backup_edge_data = graph_renderer.edge_renderer.data_source.data
    code = """
        const min_nodes = cb_obj.value[0];
        const max_nodes = cb_obj.value[1];
        var to_keep;

        to_keep = ndata.index.map((v,i) =>{
            if (ndata._type[i] !== "spin"){return i}
            else{if(ndata._num_members[i]<=max_nodes && ndata._num_members[i]>=min_nodes)
            {return i}}
        }).filter(x => Boolean(x) | x === 0);

        let keep_names = ndata.index.filter((_,i)=>to_keep.includes(i))

        let new_nodes = {};
        Object.keys(ndata).forEach(k => {
            new_nodes[k] = ndata[k].filter((_,i)=>to_keep.includes(i));
        })

        let new_edges = {};

        Object.keys(edata).forEach(k => {
            new_edges[k] = edata[k].filter((_,i)=>{
                let end = false;
                let start = false;
                if (keep_names.includes(edata.start[i]))
                    start = true
                if (keep_names.includes(edata.end[i]))
                    end = true
                return end && start;
            });
        });

        // Update graph
        graph_setup.node_renderer.data_source.data = new_nodes;
        graph_setup.edge_renderer.data_source.data = new_edges;
        graph_setup.node_renderer.data_source.change.emit();
        graph_setup.edge_renderer.data_source.change.emit();
    """
    min_cb = CustomJS(args=dict(graph_setup=graph_renderer,
                                start=df['source'].values,
                                end=df['target'].values,
                                ndata=backup_node_data,
                                edata=backup_edge_data,
                                type="min"),
                        code=code)
    min_slider = RangeSlider(title='Filter by Number of Members', start=1, end=max_members, value=(1, max_members))

    min_slider.js_on_change('value', min_cb)
    output_file(fname)
    layout = Column(plot, min_slider)
    save(layout)
    
def Bioactivity_plot(MasterOutput,Network_In_Path,Bioactivity_Data_In,title='Bioactivity_Plot',fname='Bioactivity_Plot',Bioactivity_Low=0.33,Bioactivity_Med=0.66,Bioactivity_High=1.0):
    import pandas as pd
    import os
    Bioactivity_Data = pd.read_csv(Bioactivity_Data_In,index_col=0) 
    Network_In = nx.read_graphml(Network_In_Path)
    nx.draw(Network_In,with_labels=True)
    color_map = {}
    for i in Network_In.nodes():
        if i in Bioactivity_Data.index.to_list():
            Bioactivity_Score = Bioactivity_Data.loc[i].Bioactivity_Score
            if 0<=Bioactivity_Score<Bioactivity_Low:
                color_map[i]='#EBEFBB'
            if Bioactivity_Low<=Bioactivity_Score<Bioactivity_Med:
                color_map[i]='#F9A600'
            if Bioactivity_Score>=Bioactivity_High:
                color_map[i]='#C43C00'
        else:
            pass
    nx.set_node_attributes(Network_In,color_map,'_color')
    nx.write_graphml(Network_In,str(title+'.graphml'))
    create_bokeh(Network_In, title, os.path.join(MasterOutput,fname+'.html'), extract_size=15, feature_size=10)
