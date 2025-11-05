# %%
# %%
from pptx import Presentation
from openpyxl import load_workbook
import os
from dotenv import load_dotenv
from pathlib import Path
import base64
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient
import numpy as np
import win32com.client

load_dotenv(override=True)
embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
vl_model_name = os.getenv("VL_MODEL_NAME")

text_embedder = OpenAIEmbeddings(
    model=embeddings_model_name,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
)

# DRM PPTX 처리 함수 (Windows) - 완전 자동화 버전
def export_pptx_slides_via_com_auto(
    file_path, output_dir="output_images", image_format="PNG", close_after=True
):
    """
    Windows COM을 사용하여 PowerPoint 슬라이드를 이미지로 내보내기 (완전 자동화)
    사용자가 파일을 수동으로 열 필요 없음
    DRM이 걸린 파일도 PowerPoint가 열 수 있으면 처리 가능

    Args:
        file_path: PPTX 파일 경로
        output_dir: 출력 디렉토리
        image_format: 이미지 형식 ("PNG", "JPG" 등)
        close_after: 처리 후 PowerPoint 종료 여부
    """
    if not HAS_WIN32COM:
        raise ImportError("win32com이 설치되지 않았습니다. pip install pywin32")

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir).absolute()
    filename = Path(file_path).stem
    file_path_abs = str(Path(file_path).absolute())

    ppt_app = None
    presentation = None

    try:
        # PowerPoint 애플리케이션 시작
        print("📊 PowerPoint 애플리케이션 시작 중...")
        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        ppt_app.Visible = False  # 백그라운드 실행 (선택사항: True로 하면 GUI 표시)

        # 파일이 이미 열려있는지 확인
        print(f"📂 파일 확인 중: {Path(file_path).name}")
        for i in range(ppt_app.Presentations.Count):
            pres = ppt_app.Presentations.Item(i + 1)
            if pres.FullName == file_path_abs:
                presentation = pres
                print(f"✓ 이미 열려있는 프레젠테이션 발견")
                break

        # 열려있지 않으면 새로 열기
        if presentation is None:
            print(f"📖 프레젠테이션 열기 중...")
            presentation = ppt_app.Presentations.Open(
                file_path_abs,
                ReadOnly=True,  # 읽기 전용으로 열기
                Untitled=False,  # 임시 파일이 아님
                WithWindow=False,  # 창 표시 안 함 (백그라운드)
            )
            print(f"✓ 프레젠테이션 열기 완료")

        slide_count = presentation.Slides.Count
        print(f"📊 총 {slide_count}개 슬라이드 발견")

        slide_images = []

        # 각 슬라이드를 이미지로 내보내기
        for i in range(1, slide_count + 1):
            slide = presentation.Slides.Item(i)

            # 이미지 파일 경로
            image_path = output_path / f"{filename}_slide_{i}.{image_format.lower()}"

            # 슬라이드를 이미지로 내보내기
            slide.Export(
                str(image_path),
                image_format,
                ScaleWidth=1920,  # 해상도 설정
                ScaleHeight=1080,
            )

            slide_images.append(str(image_path))
            print(f"  ✓ 슬라이드 {i}/{slide_count} 내보내기 완료")

        return slide_images

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 오류 발생: {error_msg}")

        # DRM 관련 오류 확인
        if any(
            keyword in error_msg.lower()
            for keyword in [
                "password",
                "protected",
                "locked",
                "permission",
                "access denied",
                "cannot open",
            ]
        ):
            print("\n⚠️  DRM 보호 또는 권한 문제가 감지되었습니다.")
            print("   파일을 수동으로 열 수 있는 권한이 있는지 확인해주세요.")
            raise PermissionError(f"파일 접근 권한 없음: {error_msg}")
        else:
            raise
    finally:
        # 정리 작업
        if presentation:
            try:
                if close_after:
                    presentation.Close()
                    print("✓ 프레젠테이션 닫기 완료")
            except:
                pass

        if ppt_app and close_after:
            try:
                ppt_app.Quit()
                print("✓ PowerPoint 종료 완료")
            except:
                pass


def process_drm_pptx_auto(file_path, output_dir="output_images", use_ocr=True):
    """
    DRM 보호 PPTX를 Document 객체로 변환 (완전 자동화)
    사용자가 파일을 수동으로 열 필요 없음
    """
    print(f"\n{'='*60}")
    print(f"🔒 DRM 보호 PPTX 파일 자동 처리 시작")
    print(f"{'='*60}")
    print(f"📁 파일: {Path(file_path).name}")

    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 슬라이드 이미지 추출 (완전 자동화)
    try:
        slide_images = export_pptx_slides_via_com_auto(
            file_path,
            output_dir,
            image_format="PNG",
            close_after=True,  # 처리 후 PowerPoint 종료
        )
    except PermissionError as e:
        print(f"\n❌ 파일 접근 권한 오류: {e}")
        print("\n💡 대안:")
        print("   1. 파일이 다른 프로그램에서 열려있지 않은지 확인")
        print("   2. 파일에 대한 읽기 권한이 있는지 확인")
        print("   3. 수동으로 파일을 열어주는 방법 사용 (process_drm_pptx 사용)")
        raise
    except Exception as e:
        print(f"\n❌ 자동 처리 실패: {e}")
        raise

    slide_data = []

    # 각 슬라이드 이미지 처리
    for i, img_path in enumerate(slide_images, 1):
        print(f"\n📸 슬라이드 {i}/{len(slide_images)} 처리 중...")

        # Vision 모델로 이미지 설명 생성
        print(f"  🤖 Vision 모델로 이미지 분석 중...")
        image_description = describe_image(img_path)

        # OCR로 텍스트 추출 (선택사항)
        ocr_text = ""
        if use_ocr and HAS_OCR:
            try:
                print(f"  🔍 OCR로 텍스트 추출 중...")
                ocr_text = pytesseract.image_to_string(
                    Image.open(img_path), lang="kor+eng"
                )
            except Exception as e:
                print(f"  ⚠️  OCR 실패: {e}")

        slide_data.append(
            {
                "text": ocr_text,
                "images": [img_path],
                "image_descriptions": [image_description],
                "page_number": i,
            }
        )

    # Document 객체 생성
    documents = []
    for slide in slide_data:
        content_parts = []

        if slide.get("text", "").strip():
            content_parts.append(f"[슬라이드 텍스트]\n{slide['text']}")

        if slide.get("image_descriptions"):
            for img_idx, img_desc in enumerate(slide["image_descriptions"]):
                content_parts.append(f"\n[이미지 {img_idx + 1} 설명]\n{img_desc}")

        page_content = "\n".join(content_parts)

        doc = Document(
            page_content=page_content,
            metadata={
                "source": file_path,
                "doc_type": "pptx",
                "page_number": slide["page_number"],
                "slide_text": slide.get("text", ""),
                "image_count": len(slide.get("images", [])),
                "image_paths": slide.get("images", []),
            },
        )
        documents.append(doc)

    print(f"\n✅ 처리 완료! 총 {len(documents)}개 문서 생성")
    return documents


# 6. 통합 문서 처리 및 MongoDB 저장 함수 (업데이트)
def process_and_store_document(
    file_path, output_dir="output_images", move_after_process=True, auto_open_drm=True
):
    """
    모든 타입의 문서를 처리하고 MongoDB에 저장
    DRM이 걸린 PPTX 파일도 자동으로 감지하여 처리

    Args:
        file_path: 처리할 파일 경로
        output_dir: 출력 디렉토리
        move_after_process: 처리 후 파일 이동 여부
        auto_open_drm: DRM 파일을 자동으로 열기 시도 (True) 또는 수동 안내 (False)
    """
    import shutil

    file_ext = Path(file_path).suffix.lower()
    print(f"\n{'='*60}")
    print(f"파일 처리 중: {Path(file_path).name}")
    print(f"{'='*60}")

    try:
        # 파일 타입별 처리
        if file_ext == ".pptx":
            try:
                # 일반 방법 시도
                documents = process_pptx(file_path, output_dir)
            except Exception as e:
                error_msg = str(e).lower()
                # DRM 관련 오류인지 확인
                if any(
                    keyword in error_msg
                    for keyword in [
                        "drm",
                        "protected",
                        "encrypted",
                        "permission",
                        "password",
                        "read-only",
                        "locked",
                        "cannot open",
                        "corrupted",
                        "invalid",
                        "format",
                    ]
                ):
                    print(f"\n⚠️  DRM 보호 파일 감지. 대체 방법 사용...")

                    if auto_open_drm:
                        # 완전 자동화 방법 시도
                        print("🚀 자동으로 파일을 열어 처리합니다...")
                        try:
                            documents = process_drm_pptx_auto(file_path, output_dir)
                        except PermissionError:
                            # 권한 문제 발생 시 수동 방법 안내
                            print("\n⚠️  자동 처리 실패. 수동 방법을 사용합니다...")
                            documents = process_drm_pptx(file_path, output_dir)
                    else:
                        # 수동 방법 사용
                        documents = process_drm_pptx(file_path, output_dir)
                else:
                    # 다른 오류는 그대로 전파
                    raise
        elif file_ext in [".xlsx", ".xls"]:
            documents = process_excel(file_path)
        elif file_ext in [".txt", ".pdf", ".docx", ".doc"]:
            documents = process_text_document(file_path)
        elif file_ext == ".json":
            documents = process_structured_json(file_path)
        elif file_ext in [".goodocs"]:
            documents = process_goodocs(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")

        print(f"추출된 문서 수: {len(documents)}")

        # MongoDB에 저장
        for idx, doc in enumerate(documents, 1):
            # 임베딩 생성 (대/소문자 무시 정규화 적용)
            embedding_vector = text_embedder.embed_query(
                normalize_text(doc.page_content)
            )

            # MongoDB 문서 구조
            mongo_doc = {
                "page_content": doc.page_content,
                "embedding": embedding_vector,
                "metadata": dict(doc.metadata),
            }

            # 저장
            result = collection.insert_one(mongo_doc)
            print(f"  ✓ 문서 {idx}/{len(documents)} 저장 완료")

        print(f"✅ {Path(file_path).name} 처리 완료!\n")

        # 처리 완료 후 파일 이동
        if move_after_process:
            # Complete_file 폴더 경로 설정
            source_path = Path(file_path)
            complete_folder = source_path.parent.parent / "Complete_file"

            # 폴더가 없으면 생성
            complete_folder.mkdir(parents=True, exist_ok=True)

            # 목적지 파일 경로
            destination_path = complete_folder / source_path.name

            # 같은 이름의 파일이 이미 있으면 타임스탬프 추가
            if destination_path.exists():
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = destination_path.stem
                suffix = destination_path.suffix
                destination_path = complete_folder / f"{stem}_{timestamp}{suffix}"

            # 파일 이동
            shutil.move(str(source_path), str(destination_path))
            print(f"📦 파일 이동: {destination_path}")

        return len(documents)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise


# 완전 자동화 버전 사용
files_to_process = [
    "C:/path/to/drm_protected_file.pptx",
]

for file_path in files_to_process:
    # 자동으로 파일을 열어서 처리 (사용자 개입 불필요)
    docs_count = process_and_store_document(
        file_path, move_after_process=True, auto_open_drm=True  # 자동으로 파일 열기
    )

# %%
