import 'boxicons'
function Footer() {
    const year = new Date();
  
    return (
        <div className="flex flex-col h-52 bg-slate-950 justify-center items-center  ">
            <h1 className=" text-white text-3xl ">That's all for now!</h1>
            <span className="text-sm text-gray-300 sm:text-center dark:text-gray-300">© {year.getFullYear()} Developed by <a href="https://www.linkedin.com/in/fabiano-kohut/" className="hover:underline">Fabiano Kohut.</a>
            </span>
            <p className=" text-gray-400">All Rights Reserved.</p>
            <div className="flex text-center p-1 ">
                <a target='__blank' className=' bg-white m-1 rounded h-0 ' href='https://www.linkedin.com/in/fabiano-kohut/'><box-icon name='linkedin-square' type='logo' color='#1e12ed' ></box-icon></a>
                <a target='__blank' className='bg-white m-1 rounded h-0 ' href='https://github.com/fkohut13/fkohut13'><box-icon name='github' type='logo' color='#5b6775' ></box-icon></a>
                <a target='__blank' className='bg-white m-1  rounded h-0' href='https://api.whatsapp.com/send?phone=5541998552281&text=Ola!'><box-icon type='logo' name='whatsapp' color="#128c7e"></box-icon></a>
            </div>
        </div>
        

    );
    
    
}
export default Footer;