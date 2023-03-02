import torch.nn as nn

from models.misc import build_mlp


class Adapter(nn.Module):
    def __init__(self, args, logger):
        super(Adapter, self).__init__()
        
        self.args, self.logger = args, logger
        
        assert args.adapter_refined_feat_dim == args.s3d_hidden_dim
        
        adapter_layers = []
        adapter_layers.append(
            nn.Linear(args.s3d_hidden_dim, args.bottleneck_dim))
        adapter_layers.append(
            nn.ReLU(inplace=True))
        adapter_layers.append(
            nn.Linear(args.bottleneck_dim, args.adapter_refined_feat_dim))
        self.adapter = nn.Sequential(*adapter_layers)
        
        if 'PKG' in args.adapter_objective:
            
            if 'VNM' in args.adapter_objective:
                answer_head_VNM_output_dim = args.num_nodes
                    
                self.answer_head_VNM = build_mlp(
                    input_dim=args.adapter_refined_feat_dim, 
                    hidden_dims=[answer_head_VNM_output_dim//4, 
                                 answer_head_VNM_output_dim//2],
                    output_dim=answer_head_VNM_output_dim)
                
            if 'VTM' in args.adapter_objective:
                if (args.adapter_VTM_enable_wikihow_tasks and 
                    args.adapter_VTM_enable_howto100m_tasks):
                    
                    self.answer_head_VTM = []
                    # wikihow
                    self.answer_head_VTM.append(build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.wikihow_num_tasks//2], 
                        output_dim=args.wikihow_num_tasks))
                    # howto100m
                    self.answer_head_VTM.append(build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.howto100m_num_tasks//2], 
                        output_dim=args.howto100m_num_tasks))
                    
                    self.answer_head_VTM = nn.ModuleList(self.answer_head_VTM)
                    
                elif args.adapter_VTM_enable_wikihow_tasks:
                    self.answer_head_VTM = build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.wikihow_num_tasks//2], 
                        output_dim=args.wikihow_num_tasks)
                    
                elif args.adapter_VTM_enable_howto100m_tasks:
                    self.answer_head_VTM = build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.howto100m_num_tasks//2], 
                        output_dim=args.howto100m_num_tasks)
                
            if 'TCL' in args.adapter_objective:
                if (args.adapter_TCL_enable_wikihow_tasknodes and 
                    args.adapter_TCL_enable_howto100m_tasknodes):
                    
                    self.answer_head_TCL = []
                    # wikihow
                    self.answer_head_TCL.append(build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.num_nodes//4, args.num_nodes//2], 
                        output_dim=args.num_nodes))
                    # howto100m
                    self.answer_head_TCL.append(build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.num_nodes//4, args.num_nodes//2], 
                        output_dim=args.num_nodes))
                    
                    self.answer_head_TCL = nn.ModuleList(self.answer_head_TCL)
                    
                elif args.adapter_TCL_enable_wikihow_tasknodes:
                    self.answer_head_TCL = build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.num_nodes//4, args.num_nodes//2], 
                        output_dim=args.num_nodes)
                    
                elif args.adapter_TCL_enable_howto100m_tasknodes:
                    self.answer_head_TCL = build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[args.num_nodes//4, args.num_nodes//2], 
                        output_dim=args.num_nodes)
                
            if 'NRL' in args.adapter_objective:
                answer_head_NRL_output_dim = args.num_nodes
                    
                self.answer_head_NRL = []
                for _ in range(2*args.pretrain_khop):
                    self.answer_head_NRL.append(build_mlp(
                        input_dim=args.adapter_refined_feat_dim, 
                        hidden_dims=[answer_head_NRL_output_dim//4,
                                     answer_head_NRL_output_dim//2], 
                        output_dim=answer_head_NRL_output_dim))
                self.answer_head_NRL = nn.ModuleList(self.answer_head_NRL)
            
                
        else:
            self.answer_head = build_mlp(
                input_dim=args.adapter_refined_feat_dim, 
                hidden_dims=[args.adapter_num_classes//6, 
                             args.adapter_num_classes//4, 
                             args.adapter_num_classes//2], 
                output_dim=args.adapter_num_classes)
                
            
    def forward(self, segment_feat, prediction=True):
        """
        - segment_feat: (B, 512)
        """
        refined_segment_feat = self.adapter(segment_feat) 
        
        if not prediction:
            return refined_segment_feat
    
        else:
            if 'PKG' in self.args.adapter_objective:
                if self.args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
                    
                    # VNM
                    VNM_answer = self.answer_head_VNM(refined_segment_feat)
                
                    # VTM
                    if (self.args.adapter_VTM_enable_wikihow_tasks and 
                        self.args.adapter_VTM_enable_howto100m_tasks):
                        
                        VTM_answer = []
                        for i in range(len(self.answer_head_VTM)):
                            VTM_answer.append(
                                self.answer_head_VTM[i](refined_segment_feat))
                    else:
                        VTM_answer = self.answer_head_VTM(refined_segment_feat)
                           
                    # TCL
                    if (self.args.adapter_TCL_enable_wikihow_tasknodes and 
                        self.args.adapter_TCL_enable_howto100m_tasknodes):
                        
                        TCL_answer = []
                        for i in range(len(self.answer_head_TCL)):
                            TCL_answer.append(
                                self.answer_head_TCL[i](refined_segment_feat))
                    else:
                        TCL_answer = self.answer_head_TCL(refined_segment_feat)
                       
                    # NRL
                    NRL_answer = []
                    for i in range(len(self.answer_head_NRL)):
                        NRL_answer.append(
                            self.answer_head_NRL[i](refined_segment_feat))
                    return VNM_answer, VTM_answer, TCL_answer, NRL_answer
                    
                
            else:
                return self.answer_head(refined_segment_feat)
            
        
        