from benchopt import run_benchmark

run_benchmark(
    benchmark_path="metabci\\benchmarkGUI\\benchmark",
    solver_names=[
        # "SSVEP-docomposition-algo[model=[ECCA,FBECCA,SCCA,FBSCCA,ItCCA,FBItCCA,TtCCA,FBTtCCA,MsetCCA,FBMsetCCA,MsetCCAR,FBMsetCCAR,TDCA,FBTDCA,TRCA,FBTRCA,TRCAR,FBTRCAR,],module_name=decomposition, padding_len=0]",  # ECCA,FBECCA,SCCA,FBSCCA,ItCCA,FBItCCA,TtCCA,FBTtCCA,MsetCCA,FBMsetCCA,MsetCCAR,FBMsetCCAR,TDCA,FBTDCA,TRCA,FBTRCA,TRCAR,FBTRCAR,
        # "SSVEP-docomposition-algo[custom_model=[FBSCCA], model=None, module_name=algorithm, padding_len=None]",
        "P300-docomposition-algo[model=[STDA], module_name=decomposition]",  # DCPM,SKLDA,LDA,STDA
        # "P300-docomposition-algo[custom_model=[DCPM],model=None, module_name=algorithm]",
        # "MI-docomposition-algo[model=[CSP, FBCSP, MultiCSP, FBMultiCSP, DSP, FBDSP, SSCOR, FBSSCOR], module_name=decomposition]",  # CSP, FBCSP, MultiCSP, FBMultiCSP, DSP, FBDSP, SSCOR, FBSSCOR
        # "MI-docomposition-algo[custom_model=[DSP], model=None, module_name=algorithm]",
    ],
    dataset_names=[
        # "Wang2016[channel=occipital_9,duration=[0.8],subject=[1]]",
        # "Nakanishi2015[channel=occipital_8,duration=[3.0,0.2],subject=[1]]",
        # 'BETA[channel=occipital_9,duration=[1.8], subject=[1]]',
        "Cattan_P300[subject=[1],duration=[0.3]]", 
        "Cattan_P300[channel=occipital_8,duration=[0.3,0.4,0.5,0.6],subject=[1,2,3]]"
        # "AlexMI[duration=[3],subject=[2]]",  # AlexMI
        # 'bnci2014001[duration=[4]]',
        # "munichmi[duration=[6],subject=[1]]",
        # 'eegbci[duration=[3],subject=[1]]',
        # 'schirrmeister2017[duration=[4],subject=[1]]',
        # 'weibo2014[duration=[6],subject=[1]]',
    ],
    # max_runs=None,
    # max_runs=11,
    max_runs=4,
)
