A_1 = [
    'SLIDE_000',
    'SLIDE_001',
    'SLIDE_002',
    'SLIDE_003',
    'SLIDE_004',
    'SLIDE_005',
    'SLIDE_006',
    'SLIDE_007',
    'SLIDE_008',
    'SLIDE_009',
]
A_2 = [
    'SLIDE_010',
    'SLIDE_011',
    'SLIDE_012',
    'SLIDE_013',
    'SLIDE_014',
    'SLIDE_015',
    'SLIDE_016',
    'SLIDE_017',
    'SLIDE_018',
    'SLIDE_019',
]
A_3 =[
    'SLIDE_020',
    'SLIDE_021',
    'SLIDE_022',
    'SLIDE_023',
    'SLIDE_024',
    'SLIDE_025',
    'SLIDE_026',
    'SLIDE_027',
    'SLIDE_028',
    'SLIDE_029',
]
A_4 = [
    'SLIDE_030',
    'SLIDE_031',
    'SLIDE_032',
    'SLIDE_033',
    'SLIDE_034',
    'SLIDE_035',
    'SLIDE_036',
    'SLIDE_037',
    'SLIDE_038',
    'SLIDE_039',
]
A_5 = [
    'SLIDE_040',
    'SLIDE_041',
    'SLIDE_042',
    'SLIDE_043',
    'SLIDE_044',
    'SLIDE_045',
    'SLIDE_046',
    'SLIDE_047',
    'SLIDE_048',
    'SLIDE_049',
]

B_1 = [
    'SLIDE_100',
    'SLIDE_101',
    'SLIDE_102',
    'SLIDE_103',
    'SLIDE_104',
    'SLIDE_105',
    'SLIDE_106',
    'SLIDE_107',
    'SLIDE_108',
    'SLIDE_109',
]
B_2 = [
    'SLIDE_110',
    'SLIDE_111',
    'SLIDE_112',
    'SLIDE_113',
    'SLIDE_114',
    'SLIDE_115',
    'SLIDE_116',
    'SLIDE_117',
    'SLIDE_118',
    'SLIDE_119',
]
B_3 =[
    'SLIDE_120',
    'SLIDE_121',
    'SLIDE_122',
    'SLIDE_123',
    'SLIDE_124',
    'SLIDE_125',
    'SLIDE_126',
    'SLIDE_127',
    'SLIDE_128',
    'SLIDE_129',
]
B_4 = [
    'SLIDE_130',
    'SLIDE_131',
    'SLIDE_132',
    'SLIDE_133',
    'SLIDE_134',
    'SLIDE_135',
    'SLIDE_136',
    'SLIDE_137',
    'SLIDE_138',
    'SLIDE_139',
]
B_5 = [
    'SLIDE_140',
    'SLIDE_141',
    'SLIDE_142',
    'SLIDE_143',
    'SLIDE_144',
    'SLIDE_145',
    'SLIDE_146',
    'SLIDE_147',
    'SLIDE_148',
    'SLIDE_149',
]

C_1 = [
    'SLIDE_200',
    'SLIDE_201',
    'SLIDE_202',
    'SLIDE_203',
    'SLIDE_204',
    'SLIDE_205',
    'SLIDE_206',
    'SLIDE_207',
    'SLIDE_208',
    'SLIDE_209',
]
C_2 = [
    'SLIDE_210',
    'SLIDE_211',
    'SLIDE_212',
    'SLIDE_213',
    'SLIDE_214',
    'SLIDE_215',
    'SLIDE_216',
    'SLIDE_217',
    'SLIDE_218',
    'SLIDE_219',
]
C_3 =[
    'SLIDE_220',
    'SLIDE_221',
    'SLIDE_222',
    'SLIDE_223',
    'SLIDE_224',
    'SLIDE_225',
    'SLIDE_226',
    'SLIDE_227',
    'SLIDE_228',
    'SLIDE_229',
]
C_4 = [
    'SLIDE_230',
    'SLIDE_231',
    'SLIDE_232',
    'SLIDE_233',
    'SLIDE_234',
    'SLIDE_235',
    'SLIDE_236',
    'SLIDE_237',
    'SLIDE_238',
    'SLIDE_239',
]
C_5 = [
    'SLIDE_240',
    'SLIDE_241',
    'SLIDE_242',
    'SLIDE_243',
    'SLIDE_244',
    'SLIDE_245',
    'SLIDE_246',
    'SLIDE_247',
    'SLIDE_248',
    'SLIDE_249',
]


# slideを訓練用とテスト(valid)用に分割
def slide_split(train, test):
    # ex) train = '123', test_or_valid = '4'
    
    data_map = {}
    data_map['data1'] = [A_1, B_1, C_1]
    data_map['data2'] = [A_2, B_2, C_2]
    data_map['data3'] = [A_3, B_3, C_3]
    data_map['data4'] = [A_4, B_4, C_4]
    data_map['data5'] = [A_5, B_5, C_5]

    train_list = [i for i in train]
    train_A = []
    train_B = []
    train_C = []
    for num in train_list:
        train_A = train_A + data_map[f'data{num}'][0]
        train_B = train_B + data_map[f'data{num}'][1]
        train_C = train_C + data_map[f'data{num}'][2]

    test_list = [i for i in test]
    test_A = []
    test_B = []
    test_C = []
    for num in test_list:
        test_A = test_A + data_map[f'data{num}'][0]
        test_B = test_B + data_map[f'data{num}'][1]
        test_C = test_C + data_map[f'data{num}'][2]

    return train_A, train_B, train_C, test_A, test_B, test_C
