#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.7.1 on Mon Jul  9 16:41:13 2018
#

import wx

# begin wxGlade: dependencies
import gettext
# end wxGlade

# begin wxGlade: extracode
# end wxGlade


class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyFrame.__init__
        wx.Frame.__init__(self, *args, **kwds)
        self.notebook_1 = wx.Notebook(self, wx.ID_ANY)
        self.notebook_1_pane_1 = wx.Panel(self.notebook_1, wx.ID_ANY)
        self.button_1 = wx.Button(self.notebook_1_pane_1, wx.ID_ANY, _("New"))
        #self.tree_ctrl_1 = wx.TreeCtrl(self.notebook_1_pane_1, wx.ID_ANY, style=wx.TR_HIDE_ROOT)
        self.dir = wx.GenericDirCtrl(self.notebook_1_pane_1, -1, dir='/home/pirl/PycharmProjects/tensorflowGUI/data',
                                     style=wx.DIRCTRL_DIR_ONLY)

        self.label_1 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Dataset Spec"))
        self.label_2 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Path: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_2 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_3 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("# of Classes: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_4 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_4 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("# of Images: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_5 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_5 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Image Size: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_6 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_6 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Image Type: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_7 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_7 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Resize Transformation: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_8 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_8 = wx.StaticText(self.notebook_1_pane_1, wx.ID_ANY, _("Image Encoding: "), style=wx.ALIGN_CENTER)
        self.text_ctrl_9 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.text_ctrl_3 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, "", style=wx.TE_READONLY)
        self.text_ctrl_1 = wx.TextCtrl(self.notebook_1_pane_1, wx.ID_ANY, _("log field..\n\n\n\n"), style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.notebook_1_pane_2 = wx.Panel(self.notebook_1, wx.ID_ANY)
        self.notebook_1_pane_3 = wx.Panel(self.notebook_1, wx.ID_ANY)
        self.notebook_1_pane_4 = wx.Panel(self.notebook_1, wx.ID_ANY)

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: MyFrame.__set_properties
        self.SetTitle(_("frame_1"))
        self.label_1.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, ""))
        self.text_ctrl_1.SetBackgroundColour(wx.Colour(235, 235, 235))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MyFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.VERTICAL)
        sizer_6 = wx.BoxSizer(wx.VERTICAL)
        sizer_13 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_12 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_11 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_10 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_8 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4 = wx.BoxSizer(wx.VERTICAL)
        sizer_4.Add(self.button_1, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 0)
        sizer_4.Add(self.dir, 1, wx.ALL | wx.EXPAND, 0)
        sizer_3.Add(sizer_4, 1, wx.EXPAND, 0)
        sizer_5.Add(self.label_1, 0, 0, 0)
        sizer_7.Add(self.label_2, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_7.Add(self.text_ctrl_2, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_7, 0, wx.EXPAND, 0)
        sizer_8.Add(self.label_3, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_8.Add(self.text_ctrl_4, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_8, 0, wx.EXPAND, 0)
        sizer_9.Add(self.label_4, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_9.Add(self.text_ctrl_5, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_9, 0, wx.EXPAND, 0)
        sizer_10.Add(self.label_5, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_10.Add(self.text_ctrl_6, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_10, 0, wx.EXPAND, 0)
        sizer_11.Add(self.label_6, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_11.Add(self.text_ctrl_7, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_11, 0, wx.EXPAND, 0)
        sizer_12.Add(self.label_7, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_12.Add(self.text_ctrl_8, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_12, 0, wx.EXPAND, 0)
        sizer_13.Add(self.label_8, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_13.Add(self.text_ctrl_9, 3, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_6.Add(sizer_13, 0, wx.EXPAND, 0)
        sizer_5.Add(sizer_6, 1, 0, 0)
        sizer_3.Add(sizer_5, 2, 0, 0)
        sizer_3.Add(self.text_ctrl_3, 1, wx.EXPAND, 0)
        sizer_2.Add(sizer_3, 2, wx.EXPAND, 0)
        sizer_2.Add(self.text_ctrl_1, 1, wx.EXPAND, 0)
        self.notebook_1_pane_1.SetSizer(sizer_2)
        self.notebook_1.AddPage(self.notebook_1_pane_1, _("Datasets"))
        self.notebook_1.AddPage(self.notebook_1_pane_2, _("Models"))
        self.notebook_1.AddPage(self.notebook_1_pane_3, _("Settings"))
        self.notebook_1.AddPage(self.notebook_1_pane_4, _("Etc."))
        sizer_1.Add(self.notebook_1, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()
        # end wxGlade

# end of class MyFrame

class MyDialog(wx.Dialog):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyDialog.__init__
        wx.Dialog.__init__(self, *args, **kwds)
        self.label_9 = wx.StaticText(self, wx.ID_ANY, _("\n          What kind of job do you want to do?          \n"), style=wx.ALIGN_CENTER)
        self.button_4 = wx.Button(self, wx.ID_ANY, _("Classification"))
        self.button_5 = wx.Button(self, wx.ID_ANY, _("Detection"))
        self.button_6 = wx.Button(self, wx.ID_ANY, _("Segmentation"))

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: MyDialog.__set_properties
        self.SetTitle(_("dialog_1"))
        self.label_9.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, ""))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MyDialog.__do_layout
        sizer_14 = wx.BoxSizer(wx.VERTICAL)
        sizer_15 = wx.BoxSizer(wx.VERTICAL)
        sizer_14.Add(self.label_9, 1, wx.ALIGN_CENTER | wx.ALL, 0)
        sizer_15.Add(self.button_4, 1, wx.EXPAND, 0)
        sizer_15.Add(self.button_5, 1, wx.EXPAND, 0)
        sizer_15.Add(self.button_6, 1, wx.EXPAND, 0)
        sizer_14.Add(sizer_15, 4, wx.ALIGN_CENTER | wx.EXPAND, 0)
        self.SetSizer(sizer_14)
        sizer_14.Fit(self)
        self.Layout()
        # end wxGlade

# end of class MyDialog

class MyDialog1(wx.Dialog):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyDialog1.__init__
        wx.Dialog.__init__(self, *args, **kwds)
        self.label_10 = wx.StaticText(self, wx.ID_ANY, _("Image Type"))
        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=[_("Grayscale"), _("Color")], style=wx.CB_DROPDOWN)
        self.label_11 = wx.StaticText(self, wx.ID_ANY, _("Image Size"))
        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_CENTRE)
        self.label_12 = wx.StaticText(self, wx.ID_ANY, _(" x "), style=wx.ALIGN_CENTER)
        self.text_ctrl_11 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.label_13 = wx.StaticText(self, wx.ID_ANY, _("Resize Transformation"))
        self.combo_box_2 = wx.ComboBox(self, wx.ID_ANY, choices=[_("Crop"), _("Squash")], style=wx.CB_DROPDOWN)
        self.button_2 = wx.Button(self, wx.ID_ANY, _("See example"))
        self.label_14 = wx.StaticText(self, wx.ID_ANY, _("DB backend"))
        self.combo_box_3 = wx.ComboBox(self, wx.ID_ANY, choices=[_("LMDB"), _("Sth else...")], style=wx.CB_DROPDOWN)
        self.label_15 = wx.StaticText(self, wx.ID_ANY, _("Image Encoding"))
        self.combo_box_4 = wx.ComboBox(self, wx.ID_ANY, choices=[_("png"), _("jpg"), _("....")], style=wx.CB_DROPDOWN)
        self.label_16 = wx.StaticText(self, wx.ID_ANY, _("Dataset name"))
        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.button_3 = wx.Button(self, wx.ID_ANY, _("create"))
        self.label_17 = wx.StaticText(self, wx.ID_ANY, _("Training Images"))
        self.text_ctrl_14 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.label_18 = wx.StaticText(self, wx.ID_ANY, _("Minimum samples per class"))
        self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.label_19 = wx.StaticText(self, wx.ID_ANY, _("% for validation"))
        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.label_20 = wx.StaticText(self, wx.ID_ANY, _("Maximum Samples per Class"))
        self.text_ctrl_17 = wx.TextCtrl(self, wx.ID_ANY, "")
        self.label_21 = wx.StaticText(self, wx.ID_ANY, _("% for testing"))
        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, "")

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: MyDialog1.__set_properties
        self.SetTitle(_("Create New Image Dataset"))
        self.combo_box_1.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.combo_box_1.SetSelection(1)
        self.combo_box_2.SetSelection(-1)
        self.combo_box_3.SetSelection(0)
        self.combo_box_4.SetSelection(-1)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MyDialog1.__do_layout
        sizer_16 = wx.BoxSizer(wx.VERTICAL)
        sizer_18 = wx.BoxSizer(wx.VERTICAL)
        sizer_19 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_21 = wx.BoxSizer(wx.VERTICAL)
        sizer_35 = wx.BoxSizer(wx.VERTICAL)
        sizer_34 = wx.BoxSizer(wx.VERTICAL)
        sizer_20 = wx.BoxSizer(wx.VERTICAL)
        sizer_33 = wx.BoxSizer(wx.VERTICAL)
        sizer_32 = wx.BoxSizer(wx.VERTICAL)
        sizer_31 = wx.BoxSizer(wx.VERTICAL)
        sizer_17 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_23 = wx.BoxSizer(wx.VERTICAL)
        sizer_30 = wx.BoxSizer(wx.VERTICAL)
        sizer_29 = wx.BoxSizer(wx.VERTICAL)
        sizer_28 = wx.BoxSizer(wx.VERTICAL)
        sizer_22 = wx.BoxSizer(wx.VERTICAL)
        sizer_27 = wx.BoxSizer(wx.VERTICAL)
        sizer_25 = wx.BoxSizer(wx.VERTICAL)
        sizer_26 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_24 = wx.BoxSizer(wx.VERTICAL)
        sizer_24.Add(self.label_10, 0, wx.EXPAND, 0)
        sizer_24.Add(self.combo_box_1, 0, wx.EXPAND, 0)
        sizer_22.Add(sizer_24, 1, 0, 0)
        sizer_25.Add(self.label_11, 0, 0, 0)
        sizer_26.Add(self.text_ctrl_12, 1, wx.ALIGN_CENTER, 0)
        sizer_26.Add(self.label_12, 0, wx.ALIGN_CENTER, 0)
        sizer_26.Add(self.text_ctrl_11, 1, wx.ALIGN_CENTER, 0)
        sizer_25.Add(sizer_26, 1, 0, 0)
        sizer_22.Add(sizer_25, 1, 0, 0)
        sizer_27.Add(self.label_13, 0, 0, 0)
        sizer_27.Add(self.combo_box_2, 0, wx.EXPAND, 0)
        sizer_22.Add(sizer_27, 1, 0, 0)
        sizer_22.Add(self.button_2, 0, 0, 0)
        sizer_17.Add(sizer_22, 1, 0, 0)
        sizer_28.Add(self.label_14, 0, 0, 0)
        sizer_28.Add(self.combo_box_3, 0, wx.EXPAND, 0)
        sizer_23.Add(sizer_28, 1, 0, 0)
        sizer_29.Add(self.label_15, 0, 0, 0)
        sizer_29.Add(self.combo_box_4, 0, wx.EXPAND, 0)
        sizer_23.Add(sizer_29, 1, 0, 0)
        sizer_30.Add(self.label_16, 0, 0, 0)
        sizer_30.Add(self.text_ctrl_13, 0, wx.EXPAND, 0)
        sizer_23.Add(sizer_30, 1, 0, 0)
        sizer_23.Add(self.button_3, 0, 0, 0)
        sizer_17.Add(sizer_23, 1, 0, 0)
        sizer_16.Add(sizer_17, 1, 0, 0)
        sizer_31.Add(self.label_17, 0, 0, 0)
        sizer_31.Add(self.text_ctrl_14, 0, wx.EXPAND, 0)
        sizer_18.Add(sizer_31, 1, 0, 0)
        sizer_32.Add(self.label_18, 0, 0, 0)
        sizer_32.Add(self.text_ctrl_15, 0, wx.EXPAND, 0)
        sizer_20.Add(sizer_32, 1, 0, 0)
        sizer_33.Add(self.label_19, 0, 0, 0)
        sizer_33.Add(self.text_ctrl_16, 0, wx.EXPAND, 0)
        sizer_20.Add(sizer_33, 1, 0, 0)
        sizer_19.Add(sizer_20, 1, 0, 0)
        sizer_34.Add(self.label_20, 0, 0, 0)
        sizer_34.Add(self.text_ctrl_17, 0, wx.EXPAND, 0)
        sizer_21.Add(sizer_34, 1, 0, 0)
        sizer_35.Add(self.label_21, 0, 0, 0)
        sizer_35.Add(self.text_ctrl_18, 0, wx.EXPAND, 0)
        sizer_21.Add(sizer_35, 1, 0, 0)
        sizer_19.Add(sizer_21, 1, 0, 0)
        sizer_18.Add(sizer_19, 1, 0, 0)
        sizer_16.Add(sizer_18, 1, 0, 0)
        self.SetSizer(sizer_16)
        sizer_16.Fit(self)
        self.Layout()
        # end wxGlade

# end of class MyDialog1
class MyApp(wx.App):
    def OnInit(self):
        frame_1 = MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(frame_1)
        frame_1.Show()
        dialog = MyDialog(None, wx.ID_ANY, "")
        dialog.Show()
        dialog1 = MyDialog1(None, wx.ID_ANY, "")
        dialog1.Show()
        return True

# end of class MyApp

if __name__ == "__main__":
    gettext.install("app") # replace with the appropriate catalog name

    app = MyApp(0)
    app.MainLoop()