import os
import csv
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# ✅ กำหนดพาธของไฟล์ CSV
MERGED_CSV_PATH = os.path.join(BASE_DIR, "merged_stock_sentiment_financial.csv")
MARKETCAP_THAI_CSV = os.path.join(BASE_DIR, "Finbert", "stock_data_with_marketcap_thai.csv")
MARKETCAP_USA_CSV = os.path.join(BASE_DIR, "Finbert", "stock_data_with_marketcap_usa.csv")
STOCK_CSV_PATH = os.path.join(BASE_DIR, "Stock.csv")
STOCK_DETAIL_CSV_PATH = os.path.join(BASE_DIR, "StockDetail.csv")

# ✅ **Dictionary ของ CompanyName, Market, Sector, Industry และ Description**
company_dict = {
    "ADVANC": ("Advanced Info Service Public Company Limited", "Thailand", "Communication", "Telecom Services", "Advanced Info Service Public Company Limited operates as a telecommunications and technology company primarily in Thailand. The company operates through three segments: Mobile Phone Services, Mobile Phone and Equipment Sales, and Datanet and Broadband Services. The company offers post and prepaid, and international calls and roaming services; digital e services; and Mpay, a payment solution. It also provides high-speed internet broadband services under the AIS Fibre and 3BB brands to consumers and businesses; and technologies and solutions services, including 5G and IoT, cloud and data center, communication, data analytic and AI, and business network solutions under the AIS Business brand to enterprise customers, SMEs, and large enterprises. In addition, the company operates as a service provider of call center, cellular telephone, electronic payment, and cash card; network operator; broadcasting network and television broadcasting service channels; digital platform; online advertising and outsourced contact center; mobile contents, application, and digital marketing; telecommunications, fixed-line, and data communication network; insurance broker; training; and telecommunication service operator and internet, as well as operates space, land, building, and related facilities. Further, it distributes handset and internet equipment; offers internet data center and international telephone services; develops IT systems service provider of content aggregator and outsourcing service for billing and collection; provides internet and online domain name; and develops, distributes, and services software. The company was founded in 1986 and is based in Bangkok, Thailand."),
    "INTUCH": ("Intouch Holdings Public Company Limited", "Thailand", "Communication", "Telecom Holding", "Intouch Holdings Public Company Limited, through its subsidiaries, engages in the telecommunications and other businesses in Thailand. The company is involved in the trading and rental of telecommunication equipment and accessories. It also provides internet and media services. The company was formerly known as Shin Corporation Public Company Limited and changed its name to Intouch Holdings Public Company Limited in March 2014. The company was founded in 1983 and is headquartered in Bangkok, Thailand."),
    "TRUE": ("True Corporation Public Company Limited", "Thailand", "Communication", "Telecom Services", "True Corporation Public Company Limited, together with its subsidiaries, provides telecommunications and value-added services in Thailand. The company operates through Mobile, Pay TV, and Broadband Internet and Others segments. It offers mobile, broadband Internet, Wi-Fi, television, and digital platforms and solutions. The company is also involved in entertainment, mobile equipment lessor, program production, non-government telecommunication, artist management and related, Internet services provider and distributor, and marketing management activities. In addition, it operates news channel; and provides business solutions, online digital media services on website and telecommunication devices, distribution center services, advertising sale and agency services, wireless telecommunication services, pay television, and football club and related activities management services. Further, the company designs, develops, produces, and sells software products; and offers digital solutions, privilege and online-to-offline platforms, and hospitality technology, as well as business process outsourcing services in technical service, marketing, and customer relations. The company was formerly known as TelecomAsia Corporation Public Company Limited and changed its name to True Corporation Public Company Limited in April 2004. True Corporation Public Company Limited was incorporated in 1990 and is based in Bangkok, Thailand."),
    "DITTO": ("DITTO (Thailand) Public Company Limited", "Thailand", "Technology", "IT Solutions", "Ditto (Thailand) Public Company Limited distributes data and document management solutions in Thailand. The company rents, distributes, and services photocopiers, printers, and technology products, as well as technology engineering services for projects. It offers construction, mechanical and electrical engineering systems, and information technology services. In addition, it is involved in the mangrove reforestation concession for carbon credits. Ditto (Thailand) Public Company Limited was founded in 2013 and is headquartered in Bangkok, Thailand."),
    "DIF": ("Digital Telecommunications Infrastructure Fund", "Thailand", "Real Estate", "Infrastructure Fund", "We own or are entitled to the net revenues generated from a portfolio of 16,059 telecommunications towers comprising 9,727 towers owned by the Fund (comprising True Tower Assets and TUC Towers for Additional Investment No. 2, No.3 and No. 4) and 6,332 towers from which the Fund is entitled to the net revenue (comprising the BFKT Towers, AWC Towers, AWC Towers for Additional Investment No. 1 and No. 2), including the ownership in the certain BFKT Telecom Assets after the expiry of the HSPA Agreements and certain AWC Towers after the expiry of the AWC Leasing Agreement, Additional AWC Leasing Agreement No. 1 and Additional AWC Leasing Agreement No. 2. and FOC and Upcountry Broadband System"),
    "INSET": ("Infraset Public Company Limited", "Thailand", "Technology", "IT Infrastructure", "Infraset Public Company Limited constructs data centers, information technology system, infrastructure, and telecommunication transportation infrastructure in Thailand. The company offers consulting, survey, design, installation, and system management services, as well as maintenance and supervision services for installation of telecommunications infrastructure and network equipment. It also provides maintenance and services for structural engineering systems; application service; and cyber security software. In addition, the company engages in the sale of telecommunication and information technology system equipment, as well as trades in telecom and hardware IT equipment. Infraset Public Company Limited was incorporated in 2006 and is headquartered in Bangkok, Thailand."),
    "JMART": ("Jay Mart Public Company Limited", "Thailand", "Consumer", "Retail", "Jaymart Group Holdings Public Company Limited, through its subsidiaries, engages in the wholesale and retail of mobile phones, accessories, and gadgets in Thailand. The company operates through four segments: Trading Business, Debt Collection Business, Rental Business, and Others. It is also involved in the sales of land and houses and residential condominium; debts management and collection; property development; ecommerce business; distribution of food and beverage; sale of electrical appliances; insurance broker; elderly care school; blockchain business; consulting on business operation related information technology and e-commerce business; operation of electronic network for peer-to-peer lending platform; and provision of finance leasing and consumer lending, digital point collection, and appraisal services. In addition, the company engages in managing of rental spaces in the IT and shopping mall sectors; operation of coffee shop and restaurant under the Casa Lapin and Suki Teenoi brands; and offers J point, a system that allows customer to accumulate points based on spending. The company was formerly known as Jay Mart Public Company Limited and changed its name to Jaymart Group Holdings Public Company Limited in April 2023. Jaymart Group Holdings Public Company Limited was founded in 1988 and is based in Bangkok, Thailand."),
    "INET": ("Internet Thailand Public Company Limited", "Thailand", "Technology", "Cloud Computing", "Internet Thailand Public Company Limited, together with its subsidiaries, provides Internet access, and information and communication technology services for businesses and individuals in Thailand. It operates through two segments, Access Business and Business Solutions. The company offers INET data center solutions, including co-location and business continuity planning center; cloud solutions, such as virtual machine as a service, backup as a service, disaster recovery as a service, database as a service, hybrid cloud, SAP HANA as a services, container as a service, and web hosting services; and smart office solutions comprising file sharing as a services, desktop as a service, and document management system. It also provides NET Cloud Connect, which links the networks from different locations to the central infrastructure; INET NODE, a leased line; INET Load Balance, a system management to minimize the unnecessary expenses; and INET WebEx, an eMeeting for the efficient online meeting. In addition, the company provides security as a service, including INET VFW, a computer network security system; INET iLog, a traffic data storage on computer; and INET Hybrid WAN, a network connection from different locations to the center infrastructure. Further, it provides smart solution, such as artificial intelligence, chatbot as a service, analytics as a service, and big data as a service. The company was founded in 1995 and is based in Bangkok, Thailand."),
    "JAS": ("Jasmine International Public Company Limited", "Thailand", "Communication", "Broadband Services", "Jasmine International Public Company Limited engages in the telecommunications business in Thailand. It operates through three segments: Internet TV; Digital Asset and Technology Solution; and Other. The company provides content for internet protocol television services; satellite telecommunications services; internet and international calling card services; online movie and internet protocol television services; computer system integration, software development, and cloud computing services; and high-speed data communication services. It also offers circuit leasing, local and international data communication, and Bitcoin mining services; cloud AI, internet of things, and fintech; and engineer design and consultancy services in energy management and clean energy systems. In addition, the company engages in the rental of office building; distribution of computer products; design, installation, and testing of telecommunications systems; survey, design, and construction for civil work of telecommunications projects; and generation and distribution of electricity from renewable and all other energies. The company was founded in 1982 and is based in Nonthaburi, Thailand."),
    "HUMAN": ("Humanica Public Company Limited", "Thailand", "Technology", "HR Software", "Humanica Public Company Limited provides human resource services and solutions in Thailand, Singapore, Japan, Malaysia, Indonesia, Vietnam, Philippines, and internationally. The company offers various solutions, such as Workplaze, an employee-centric HR solution; Workplaze Time; Workplaze Talent Management; Workplaze Analytics and Reporting; Workplaze Attendance Recording; and Workplaze Mobile, as well as Enterprise Resource Planning (ERP) system implementation solutions. It also provides accounting and financial services; visa and work permit service, and company registration services; software-as-a-service (SaaS); consulting; and project implementation services. In addition, the company offers payroll outsourcing services; implementation services for human resource systems and computer software for enterprise resource planning; life and non-life insurance brokerage services. Further, it sells advance access control devices. The company was founded in 2003 and is headquartered in Bangkok, Thailand."),
    "AMD": ("Advanced Micro Devices Inc.", "America", "Technology", "Semiconductors", "Advanced Micro Devices, Inc. operates as a semiconductor company worldwide. It operates through four segments: Data Center, Client, Gaming, and Embedded. The company offers artificial intelligence (AI) accelerators, x86 microprocessors, and graphics processing units (GPUs) as standalone devices or as incorporated into accelerated processing units, chipsets, and data center and professional GPUs; and embedded processors and semi-custom system-on-chip (SoC) products, microprocessor and SoC development services and technology, data processing units, field programmable gate arrays (FPGA), system on modules, smart network interface cards, and adaptive SoC products. It provides processors under the AMD Ryzen, AMD Ryzen AI, AMD Ryzen PRO, AMD Ryzen Threadripper, AMD Ryzen Threadripper PRO, AMD Athlon, and AMD PRO A-Series brands; graphics under the AMD Radeon graphics and AMD Embedded Radeon graphics; and professional graphics under the AMD Radeon Pro graphics brand. The company offers data center graphics under the AMD Instinct accelerators and Radeon PRO V-series brands; server microprocessors under the AMD EPYC brand; low power solutions under the AMD Athlon, AMD Geode, AMD Ryzen, AMD EPYC, and AMD R-Series and G-Series brands; FPGA products under the Virtex-6, Virtex-7, Virtex UltraScale+, Kintex-7, Kintex UltraScale, Kintex UltraScale+, Artix-7, Artix UltraScale+, Spartan-6, and Spartan-7 brands; adaptive SOCs under the Zynq-7000, Zynq UltraScale+ MPSoC, Zynq UltraScale+ RFSoCs, Versal HBM, Versal Premium, Versal Prime, Versal AI Core, Versal AI Edge, Vitis, and Vivado brands; and compute and network acceleration board products under the Alveo and Pensando brands. It serves original equipment and design manufacturers, public cloud service providers, system integrators, independent distributors, and add-in-board manufacturers through its direct sales force and sales representatives. The company was incorporated in 1969 and is headquartered in Santa Clara, California."),
    "TSM": ("Taiwan Semiconductor Manufacturing Company", "America", "Technology", "Semiconductors", "Taiwan Semiconductor Manufacturing Company Limited, together with its subsidiaries, manufactures, packages, tests, and sells integrated circuits and other semiconductor devices in Taiwan, China, Europe, the Middle East, Africa, Japan, the United States, and internationally. It provides various wafer fabrication processes, such as processes to manufacture complementary metal- oxide-semiconductor (CMOS) logic, mixed-signal, radio frequency, embedded memory, bipolar CMOS mixed-signal, and others. The company also offers customer and engineering support services; manufactures masks; and invests in technology start-up companies; researches, designs, develops, manufactures, packages, tests, and sells color filters; and provides investment services. Its products are used in high performance computing, smartphones, Internet of things, automotive, and digital consumer electronics. The company was incorporated in 1987 and is headquartered in Hsinchu City, Taiwan."),
    "AVGO": ("Broadcom Inc.", "America", "Technology", "Semiconductors", "Broadcom Inc. designs, develops, and supplies various semiconductor devices with a focus on complex digital and mixed signal complementary metal oxide semiconductor based devices and analog III-V based products worldwide. The company operates in two segments, Semiconductor Solutions and Infrastructure Software. It provides Ethernet switching and routing custom silicon solutions, optical and copper physical layer devices, and fiber optic transmitter and receiver components; set-top box system-on-chips (SoCs), data over cable service interface specifications cable modem and networking infrastructure, DSL access multiplexer/PON optical line termination products, and Wi-Fi access point SoCs, as well as digital subscriber line (DSL)/cable, passive optical networking (PON) gateways; and fiber optic components and mobile device connectivity solutions. The company also offers RF front end modules and filters; Wi-Fi, Bluetooth, and global positioning system/global navigation satellite system SoCs; custom touch controllers; inductive charging application specific integrated circuits; serial attached small computer system interface and redundant array of independent disks controllers and adapters, peripheral component interconnect express switches, fiber channel host bus adapters, read channel based SoCs, custom flash controllers, and preamplifiers; and optocouplers, industrial fiber optics, industrial and medical sensors, motion control encoders and subsystems, light emitting diodes, ethernet PHYs, switch ICs, and camera microcontrollers. Its products are used in various applications in enterprise and data center networking, including artificial intelligence networking and connectivity, home connectivity, set-top boxes, broadband access, telecommunication equipment, smartphones and base stations, data center servers and storage systems, factory automation, power generation and alternative energy systems, and electronic displays. Broadcom Inc. was founded in 1961 and is headquartered in Palo Alto, California."),
    "TSLA": ("Tesla Inc.", "America", "Consumer", "Electric Vehicles", "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally. The company operates in two segments, Automotive; and Energy Generation and Storage. The Automotive segment offers electric vehicles, as well as sells automotive regulatory credits; and non-warranty after-sales vehicle, used vehicles, body shop and parts, supercharging, retail merchandise, and vehicle insurance services. This segment also provides sedans and sport utility vehicles through direct and used vehicle sales, a network of Tesla Superchargers, and in-app upgrades; purchase financing and leasing services; services for electric vehicles through its company-owned service locations and Tesla mobile service technicians; and vehicle limited warranties and extended service plans. The Energy Generation and Storage segment engages in the design, manufacture, installation, sale, and leasing of solar energy generation and energy storage products, and related services to residential, commercial, and industrial customers and utilities through its website, stores, and galleries, as well as through a network of channel partners. This segment also provides services and repairs to its energy product customers, including under warranty; and various financing options to its residential customers. The company was formerly known as Tesla Motors, Inc. and changed its name to Tesla, Inc. in February 2017. Tesla, Inc. was incorporated in 2003 and is headquartered in Austin, Texas."),
    "META": ("Meta Platforms Inc.", "America", "Technology", "Social Media", "Meta Platforms, Inc. engages in the development of products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality and mixed reality headsets, augmented reality, and wearables worldwide. It operates through two segments, Family of Apps (FoA) and Reality Labs (RL). The FoA segment offers Facebook, which enables people to build community through feed, reels, stories, groups, marketplace, and other; Instagram that brings people closer through instagram feed, stories, reels, live, and messaging; Messenger, a messaging application for people to connect with friends, family, communities, and businesses across platforms and devices through text, audio, and video calls; Threads, an application for text-based updates and public conversations; and WhatsApp, a messaging application that is used by people and businesses to communicate and transact in a private way. The RL segment provides virtual, augmented, and mixed reality related products comprising consumer hardware, software, and content that help people feel connected, anytime, and anywhere. The company was formerly known as Facebook, Inc. and changed its name to Meta Platforms, Inc. in October 2021. The company was incorporated in 2004 and is headquartered in Menlo Park, California."),
    "GOOGL": ("Alphabet Inc. (Google)", "America", "Technology", "Internet Services", "Alphabet Inc. offers various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments. The Google Services segment provides products and services, including ads, Android, Chrome, devices, Gmail, Google Drive, Google Maps, Google Photos, Google Play, Search, and YouTube. It is also involved in the sale of apps and in-app purchases and digital content in the Google Play and YouTube; and devices, as well as in the provision of YouTube consumer subscription services. The Google Cloud segment offers AI infrastructure, Vertex AI platform, cybersecurity, data and analytics, and other services; Google Workspace that include cloud-based communication and collaboration tools for enterprises, such as Calendar, Gmail, Docs, Drive, and Meet; and other services for enterprise customers. The Other Bets segment sells healthcare-related and internet services. The company was incorporated in 1998 and is headquartered in Mountain View, California."),
    "AMZN": ("Amazon.com Inc.", "America", "Consumer", "E-Commerce", "แพลตฟAmazon.com, Inc. engages in the retail sale of consumer products, advertising, and subscriptions service through online and physical stores in North America and internationally. The company operates through three segments: North America, International, and Amazon Web Services (AWS). It also manufactures and sells electronic devices, including Kindle, fire tablets, fire TVs, echo, ring, blink, and eero; and develops and produces media content. In addition, the company offers programs that enable sellers to sell their products in its stores; and programs that allow authors, independent publishers, musicians, filmmakers, Twitch streamers, skill and app developers, and others to publish and sell content. Further, it provides compute, storage, database, analytics, machine learning, and other services, as well as advertising services through programs, such as sponsored ads, display, and video advertising. Additionally, the company offers Amazon Prime, a membership program. The company's products offered through its stores include merchandise and content purchased for resale and products offered by third-party sellers. It serves consumers, sellers, developers, enterprises, content creators, advertisers, and employees. Amazon.com, Inc. was incorporated in 1994 and is headquartered in Seattle, Washington.อร์มอีคอมเมิร์ซและคลาวด์คอมพิวติ้ง"),
    "NVDA": ("NVIDIA Corporation", "America", "Technology", "Semiconductors", "NVIDIA Corporation, a computing infrastructure company, provides graphics and compute and networking solutions in the United States, Singapore, Taiwan, China, Hong Kong, and internationally. The Compute & Networking segment comprises Data Center computing platforms and end-to-end networking platforms, including Quantum for InfiniBand and Spectrum for Ethernet; NVIDIA DRIVE automated-driving platform and automotive development agreements; Jetson robotics and other embedded platforms; NVIDIA AI Enterprise and other software; and DGX Cloud software and services. The Graphics segment offers GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service and related infrastructure, and solutions for gaming platforms; Quadro/NVIDIA RTX GPUs for enterprise workstation graphics; virtual GPU or vGPU software for cloud-based visual and virtual computing; automotive platforms for infotainment systems; and Omniverse software for building and operating industrial AI and digital twin applications. The company's products are used in gaming, professional visualization, data center, and automotive markets. It sells its products to original equipment manufacturers, original device manufacturers, system integrators and distributors, independent software vendors, cloud service providers, consumer internet companies, add-in board manufacturers, distributors, automotive manufacturers and tier-1 automotive suppliers, and other ecosystem participants. NVIDIA Corporation was incorporated in 1993 and is headquartered in Santa Clara, California."),
    "MSFT": ("Microsoft Corporation", "America", "Technology", "Software", "Microsoft Corporation develops and supports software, services, devices and solutions worldwide. The Productivity and Business Processes segment offers office, exchange, SharePoint, Microsoft Teams, office 365 Security and Compliance, Microsoft viva, and Microsoft 365 copilot; and office consumer services, such as Microsoft 365 consumer subscriptions, Office licensed on-premises, and other office services. This segment also provides LinkedIn; and dynamics business solutions, including Dynamics 365, a set of intelligent, cloud-based applications across ERP, CRM, power apps, and power automate; and on-premises ERP and CRM applications. The Intelligent Cloud segment offers server products and cloud services, such as azure and other cloud services; SQL and windows server, visual studio, system center, and related client access licenses, as well as nuance and GitHub; and enterprise services including enterprise support services, industry solutions, and nuance professional services. The More Personal Computing segment offers Windows, including windows OEM licensing and other non-volume licensing of the Windows operating system; Windows commercial comprising volume licensing of the Windows operating system, windows cloud services, and other Windows commercial offerings; patent licensing; and windows Internet of Things; and devices, such as surface, HoloLens, and PC accessories. Additionally, this segment provides gaming, which includes Xbox hardware and content, and first- and third-party content; Xbox game pass and other subscriptions, cloud gaming, advertising, third-party disc royalties, and other cloud services; and search and news advertising, which includes Bing, Microsoft News and Edge, and third-party affiliates. The company sells its products through OEMs, distributors, and resellers; and directly through digital marketplaces, online, and retail stores. The company was founded in 1975 and is headquartered in Redmond, Washington."),
    "AAPL": ("Apple Inc.", "America", "Technology", "Consumer Electronics", "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discover and download applications and digital content, such as books, music, video, games, and podcasts, as well as advertising services include third-party licensing arrangements and its own advertising platforms. In addition, the company offers various subscription-based services, such as Apple Arcade, a game subscription service; Apple Fitness+, a personalized fitness service; Apple Music, which offers users a curated listening experience with on-demand radio stations; Apple News+, a subscription news and magazine service; Apple TV+, which offers exclusive original content; Apple Card, a co-branded credit card; and Apple Pay, a cashless payment service, as well as licenses its intellectual property. The company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. It distributes third-party applications for its products through the App Store. The company also sells its products through its retail and online stores, and direct sales force; and third-party cellular network carriers, wholesalers, retailers, and resellers. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California."),
}

# ✅ โหลดข้อมูลจาก MarketCap CSV
def load_marketcap_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={"Ticker": "StockSymbol", "Market Cap": "MarketCap"})
    # ✅ คำนวณ Change (%) จาก Open และ Close
    df["Change"] = df["Close"] - df["Open"]
    df["Change (%)"] = (df["Change"] / df["Open"]) * 100

    # ✅ เลือกเฉพาะคอลัมน์ที่จำเป็น
    df = df[["StockSymbol", "Date", "MarketCap", "Change (%)"]]

    return df

marketcap_thai_df = load_marketcap_data(MARKETCAP_THAI_CSV)
marketcap_usa_df = load_marketcap_data(MARKETCAP_USA_CSV)

# ✅ รวม MarketCap จากไทยและอเมริกา
marketcap_df = pd.concat([marketcap_thai_df, marketcap_usa_df])

# ✅ โหลดข้อมูลจาก merged_stock_sentiment_financial.csv
print("📥 กำลังโหลดไฟล์ merged_stock_sentiment_financial.csv ...")
df = pd.read_csv(MERGED_CSV_PATH)

# ✅ แปลงชื่อ Column ให้ตรงกับ Database
df = df.rename(columns={
    "Ticker": "StockSymbol",
    "Open": "OpenPrice",
    "High": "HighPrice",
    "Low": "LowPrice",
    "Close": "ClosePrice",
    "P/E Ratio ": "PERatio",
    "ROE (%)": "ROE",
    "Dividend Yield (%)": "DividendYield",
    "QoQ Growth (%)": "QoQGrowth",
    "YoY Growth (%)": "YoYGrowth",
    "Total Revenue": "TotalRevenue",
    "Net Profit": "NetProfit",
    "Earnings Per Share (EPS)": "EPS",
    "Gross Margin (%)": "GrossMargin",
    "Net Profit Margin (%)": "NetProfitMargin",
    "Debt to Equity ": "DebtToEquity"
})

# ✅ เติมข้อมูล CompanyName, Market, Sector, Industry, Description
df["CompanyName"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[0])
df["Market"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[1])
df["Sector"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[2])
df["Industry"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[3])
df["Description"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[4])


# ✅ ผสม MarketCap ให้ตรงกับ StockSymbol และ Date
df = df.merge(marketcap_df[["StockSymbol", "Date", "MarketCap", "Change (%)"]], 
              on=["StockSymbol", "Date"], how="left")


# ✅ จัดการค่า NaN เพื่อให้สอดคล้องกับ Database
df = df.where(pd.notna(df), None)  # แปลง NaN เป็น None เพื่อใช้กับ MySQL

# ✅ แยกข้อมูลสำหรับ Stock และเพิ่ม Sector, Industry, Description
stock__data = df[["StockSymbol", "Market", "MarketCap", "CompanyName", "Sector", "Industry", "Description"]].drop_duplicates(subset=["StockSymbol"], keep="last")
stock_detail_data = df[[
    "Date", "StockSymbol", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "PERatio", "ROE", "DividendYield",
    "QoQGrowth", "YoYGrowth", "TotalRevenue", "NetProfit", "EPS", "GrossMargin", "NetProfitMargin", "DebtToEquity",
    "Change (%)", "Volume"  # ✅ เพิ่ม Change (%) และ Volume
]]

print(stock__data.head(10))  # แสดง 10 แถวแรกของ Stock ก่อนบันทึกลงฐานข้อมูล

stock_detail_data = stock_detail_data.copy()  # ✅ สร้าง Copy ป้องกัน Warning
stock_detail_data.loc[:, "PredictionTrend"] = None
stock_detail_data.loc[:, "PredictionClose"] = None


# ✅ บันทึกไฟล์ CSV แยกกัน
print(f"💾 กำลังบันทึกไฟล์ {STOCK_CSV_PATH} ...")
stock__data.to_csv(STOCK_CSV_PATH, index=False, na_rep="NULL")
print(f"✅ บันทึกไฟล์ {STOCK_CSV_PATH} สำเร็จ!")

print(f"💾 กำลังบันทึกไฟล์ {STOCK_DETAIL_CSV_PATH} ...")
stock_detail_data.to_csv(STOCK_DETAIL_CSV_PATH, index=False, na_rep="NULL")
print(f"✅ บันทึกไฟล์ {STOCK_DETAIL_CSV_PATH} สำเร็จ!")

# ✅ เชื่อมต่อฐานข้อมูล
try:
    print("🔗 กำลังเชื่อมต่อกับฐานข้อมูล ...")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True
    )
    cursor = conn.cursor()
    print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

    # ✅ ฟังก์ชันแปลง NaN เป็น None ก่อน Insert
    def convert_nan_to_none(data_list):
        return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

    # ✅ **บันทึกข้อมูลลง Stock**
    insert_stock_query = """
    INSERT INTO Stock (StockSymbol, Market, MarketCap, CompanyName, Sector, Industry, Description)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        Market=COALESCE(VALUES(Market), Market),
        MarketCap=COALESCE(VALUES(MarketCap), MarketCap),
        CompanyName=COALESCE(VALUES(CompanyName), CompanyName),
        Sector=COALESCE(VALUES(Sector), Sector),
        Industry=COALESCE(VALUES(Industry), Industry),
        Description=COALESCE(VALUES(Description), Description);
    """

    # ✅ ตรวจสอบค่าก่อน insert
    print("🔍 ตัวอย่างข้อมูลที่กำลังจะบันทึกลง Stock:")
    print(stock__data.head(5).to_string())  # แสดง 5 แถวแรกก่อนบันทึก

    stock_values = convert_nan_to_none(stock__data.values.tolist())
    cursor.executemany(insert_stock_query, stock_values)
    print(f"✅ บันทึกข้อมูลลง Stock: {len(stock_values)} รายการ")

    # ✅ แปลงค่า Volume เป็น int ก่อนบันทึกลงฐานข้อมูล
    stock_detail_data["Volume"] = stock_detail_data["Volume"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    # ✅ **บันทึกข้อมูลลง StockDetail**
    insert_stock_detail_query = """
    INSERT INTO StockDetail (
        Date, StockSymbol, OpenPrice, HighPrice, LowPrice, ClosePrice, PERatio, ROE, DividendYield,
        QoQGrowth, YoYGrowth, TotalRevenue, NetProfit, EPS, GrossMargin, NetProfitMargin, DebtToEquity,
        `Change (%)`, Volume, PredictionTrend, PredictionClose
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        OpenPrice=VALUES(OpenPrice), HighPrice=VALUES(HighPrice), LowPrice=VALUES(LowPrice), 
        ClosePrice=VALUES(ClosePrice), PERatio=VALUES(PERatio), ROE=VALUES(ROE), DividendYield=VALUES(DividendYield),
        QoQGrowth=VALUES(QoQGrowth), YoYGrowth=VALUES(YoYGrowth), TotalRevenue=VALUES(TotalRevenue), 
        NetProfit=VALUES(NetProfit), EPS=VALUES(EPS), GrossMargin=VALUES(GrossMargin), 
        NetProfitMargin=VALUES(NetProfitMargin), DebtToEquity=VALUES(DebtToEquity),
        `Change (%)`=VALUES(`Change (%)`), Volume=VALUES(Volume),
        PredictionTrend=VALUES(PredictionTrend), PredictionClose=VALUES(PredictionClose);
    """
    
    stock_detail_values = convert_nan_to_none(stock_detail_data.values.tolist())
    cursor.executemany(insert_stock_detail_query, stock_detail_values)
    print(f"✅ บันทึกข้อมูลลง StockDetail: {len(stock_detail_values)} รายการ")

except mysql.connector.Error as err:
    print(f"❌ เกิดข้อผิดพลาด: {err}")

finally:
    cursor.close()
    conn.close()
    print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
