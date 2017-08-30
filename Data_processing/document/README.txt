1. # python check_file_name.py
  --> 중복을 제거한 태그 이름을 가져옵니다.
  --> 확인용. Initialize.py 에서 분할된 파일을 직접 명시하지 않으면,
      자동으로 실행
  --> ./filelist/filelist.txt 에 결과 생성

2. # python initialize.py -h
  --> 입력하고자 하는 인자에 대한 도움말이 나옵니다.

3. 인자 입력 방법
  --> 
      # python initialize.py --datadir(데이터가 저장된 폴더) ./raw_data \
      > --save_dir(RTDB가 저장될 폴더) ./RTDB \
      > --start_date 2017-01-01(원하는 시작일) \
      > --end_data 2017-07-07(원하는 마감일) \
      > --rm_old_merge(분할된 파일을 병합한 이전 파일 삭제 유무) No(default)
      > --split_files 700TI034 700PIC005B 700AI006 
	(default 값은 None, None으로 되어있을 시 check_filename 모듈에 의해
	 태그 이름 확인 후 자동으로 merge) 

4. initialize.py 코드에서

	parser.add_argument(
	'--split_data',

	부분에서

	default = check_file_name.py의 결과인 filelist
		   폴더의 filelist.txt의 'The number of file' 을 제외한
		   위의 문자열을 copy & paste한다


---------위의 인자들은 고정 시 initialize.py의 default 값을 변경하면 된다.-----


5. DB 설정 방법
  --> initialize.py의 import 아래
  --> id, password, host, database에 맞게 설정

  --> 현재는 주석 처리
