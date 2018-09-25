




from MyDemo.My.Model import Model
import time
import wx


WINDOW = (600, 600)  #窗口默认大小
COLORS = [
    wx.RED,
]


class Panel(wx.Panel):
    
    def __init__(self, parent):
        super(Panel, self).__init__(parent)
        self.model = Model(WINDOW[0], WINDOW[1])
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)   #按鼠标左键
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)  #鼠标右键
        
        self.timestamp = time.time()  #获取当前时间，时间戳
        self.on_timer()
    
    def on_timer(self):  #更新界面，每一段时间刷新一次
        now = time.time()  #返回当前时间的时间戳（1970纪元后经过的浮点秒数）
        dt = now - self.timestamp  #从初始化到现在经过了多长时间
        self.timestamp = now
        
        self.model.update(dt)  #更新界面，每隔一段时间就更新数据和界面
        self.Refresh()
        wx.CallLater(20, self.on_timer)
    
    def on_left_down(self, event):  #鼠标左键
        self.model.player.target = event.GetPosition()
    def on_right_down(self, event):  #鼠标右键  游戏重新来过
        width, height = self.GetClientSize()
        # self.model = Model(width, height)
        self.model.reset(width, height)
    def on_size(self, event):  #界面大小变化时
        width, height = self.GetClientSize()
        # self.model = Model(width, height)
        self.model.reset(width, height)
        event.Skip()
        self.Refresh()
    
    def on_paint(self, event):
        n = len(COLORS)
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.BLACK_BRUSH)
        dc.Clear()
        
        dc.SetPen(wx.BLACK_PEN)  #后面如果不是第一个，就绘制成黑色的历史轨迹，就看不见了
        for index, bot in enumerate(self.model.bots[:n]):  #返回索引和对象
            dc.SetBrush(wx.Brush(COLORS[index]))
            #绘制历史轨迹
            for x, y in bot.history:
                dc.DrawCircle(x, y, 2)
        
        dc.SetBrush(wx.BLACK_BRUSH)  #这种只画边线
        for index, bot in enumerate(self.model.bots[:n]):  #绘制每个点的目标位置
            dc.SetPen(wx.Pen(COLORS[index]))
            x, y = bot.target
            dc.DrawCircle(x, y, 6)
        
        for index, bot in enumerate(self.model.bots):
            dc.SetPen(wx.BLACK_PEN)
            if index < n:
                dc.SetBrush(wx.Brush(COLORS[index]))
            else:
                dc.SetBrush(wx.WHITE_BRUSH)
            
            x, y = bot.position
            dc.DrawCircle(x, y, 6)


class Frame(wx.Frame):
    def __init__(self):
        super(Frame, self).__init__(None)
        self.SetTitle('Motion')
        self.SetClientSize(WINDOW)
        Panel(self)  #传入的是父级组建Fram，再Pane参数是parent

def main():
    app = wx.App()
    frame = Frame()
    frame.Center()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()









